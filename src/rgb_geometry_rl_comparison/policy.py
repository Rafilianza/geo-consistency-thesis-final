"""Historical R/S/L architecture and sampler with four adapted learning rules.

All videos use the campaign's fifty Euler steps, four Gaussian actions and
frozen semantic-flow observer. Advantages, losses, replay passes and learning
rates remain specific to this learning suite. Flash/Flow here are sampler
adaptations, not reproductions of their published SDE protocols. No policy
checkpoint is written.
"""

from contextlib import contextmanager
import copy
import math
import time

import torch

from rgb_geometry_campaign import policy as campaign
from rgb_geometry_pilot import policy as wan
from rgb_geometry_rl_methods import objectives


ACTION_STEPS = campaign.ACTION_STEPS
INFERENCE_STEPS = campaign.INFERENCE_STEPS
RGB_SHAPE = campaign.RGB_SHAPE
SEMANTIC_SHAPE = campaign.SEMANTIC_SHAPE
METHODS = ('flash_grpo', 'awm', 'diffusion_nft', 'flow_grpo')


def initialize(runtime, args, config, spec):
    """Keep the already validated historical observer; install a fresh R/S/L arm."""
    arm = spec['arm']
    campaign.initialize(runtime, arm, seed=config['settings']['seed'])
    runtime['spec'] = dict(spec)
    if spec['method'] == 'diffusion_nft':
        runtime['collection_state'] = state_dict(runtime)


def parameters(runtime):
    return campaign.parameters(runtime)


def make_optimizer(runtime, config):
    """Learning rates belong to this objective suite, not historical RLOO."""
    name = 'lora_learning_rate' if runtime['arm'] == 'L' else 'head_learning_rate'
    return torch.optim.Adam(parameters(runtime), lr=config['settings'][name],
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0)


def state_dict(runtime):
    """Small, flat CPU parameter snapshots for disposable probes and NFT EMA."""
    values = campaign.state_dict(runtime)['parameters']
    return {name: value.detach().float().cpu().clone() for name, value in values.items()}


def load_state_dict(runtime, values):
    campaign.load_state_dict(runtime, {'arm': runtime['arm'],
        'parameter_version': runtime['parameter_version'], 'parameters': values})


def policy_digest(runtime):
    from rgb_geometry_campaign.protocol import tensor_digest
    return tensor_digest(state_dict(runtime))


@contextmanager
def temporary_policy(runtime, values):
    current = state_dict(runtime)
    try:
        load_state_dict(runtime, values)
        yield
    finally:
        load_state_dict(runtime, current)


def prepare(runtime):
    pipeline, device = runtime['pipeline'], runtime['device']
    pipeline.vae.cpu()
    pipeline.text_encoder.cpu()
    runtime['codec'].cpu()
    if 'teacher' in runtime:
        runtime['teacher'].to('cpu')
    pipeline.transformer.to(device).eval()
    runtime['semantic'].to(device).eval().requires_grad_(False)
    if 'head' in runtime:
        runtime['head'].to(device=device, dtype=torch.float32).eval()
    if getattr(pipeline.config, 'expand_timesteps', False):
        raise ValueError('Only scalar-timestep Wan T2V is supported.')


def schedule_for(runtime, steps):
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    if steps != INFERENCE_STEPS:
        raise ValueError('Historical matching requires fifty denoising steps.')
    scheduler = UniPCMultistepScheduler.from_config(runtime['pipeline'].scheduler.config)
    scheduler.set_timesteps(steps, device=runtime['device'])
    if not scheduler.config.use_flow_sigmas or float(scheduler.config.flow_shift) != 3:
        raise ValueError('Expected the historical flow-sigma clock with shift three.')
    if (len(scheduler.sigmas) != steps + 1 or float(scheduler.sigmas[-1]) != 0
            or not bool((scheduler.sigmas[:-1] > scheduler.sigmas[1:]).all())):
        raise ValueError('The clock must have descending sigmas and terminal zero.')
    return scheduler


@torch.no_grad()
def context_at(runtime, state, semantic_state, sigma, timestep, prompt):
    """Read frozen Wan features and predict clean semantics before the RGB action."""
    frozen_flow, features = campaign.capture_base(runtime, state, timestep, prompt, (15,))
    device = state.device
    timestep = torch.as_tensor(timestep, device=device).float()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        semantic_flow = runtime['semantic'](semantic_state.float(), features[15], timestep.reshape(1)).float()
    q = semantic_state.float() - float(sigma) * semantic_flow
    if not bool(torch.isfinite(q).all() and torch.isfinite(semantic_flow).all()):
        raise FloatingPointError('Nonfinite frozen historical semantic prediction.')
    if runtime['arm'] == 'R':
        q = torch.zeros_like(q)
    context = {'state': state.detach(), 'frozen_flow': frozen_flow.detach(),
               'q': q.detach(), 'timestep': timestep, 'prompt': prompt}
    return context, semantic_flow.detach()


def velocity(runtime, context, sigma, checkpointed=False):
    """R/S train only the head; L trains only the historical rank-four adapters."""
    base = context['frozen_flow'].detach().float()
    delta = campaign._raw_correction(runtime, context['state'].detach(), base,
        context['q'].detach(), sigma, context['timestep'], context['prompt'], checkpointed)
    correction, diagnostics = campaign.bounded_correction(delta, base, alpha=0.25)
    return base + correction, diagnostics


def transition(state, flow, step_size):
    """The historical action law: Euler mean plus 0.03*sqrt(h) Gaussian noise."""
    if not math.isfinite(float(step_size)) or float(step_size) <= 0:
        raise ValueError('The historical Euler step must be positive and finite.')
    with torch.autocast(device_type=state.device.type, enabled=False):
        mean = state.detach().float() - float(step_size) * flow.float()
    return mean, 0.03 * math.sqrt(float(step_size))


def flash_weights(sigmas):
    """Inverse mean-to-noise sensitivity; normalize over all four possible steps.

    Euler has h/std = sqrt(h)/0.03, so its inverse is 0.03/sqrt(h).
    Uniformly sampling one position with this fixed normalization estimates the
    full four-position weighted mean without a sample-dependent denominator.
    """
    weights = {}
    for step_index in ACTION_STEPS:
        step_size = float(sigmas[step_index] - sigmas[step_index + 1])
        weights[step_index] = 0.03 / math.sqrt(step_size)
    average = sum(weights.values()) / len(weights)
    return {step_index: weight / average for step_index, weight in weights.items()}


def _cpu_context(context):
    return {name: value.detach().float().cpu().clone() if torch.is_tensor(value) else value
            for name, value in context.items()}


def _device_context(runtime, context):
    return {name: value.to(runtime['device']) if torch.is_tensor(value) else value
            for name, value in context.items()}


@torch.no_grad()
def collect_trajectory(runtime, case, method, steps, training):
    """The method changes replay selection, never the generation noise schedule."""
    if training and method == 'diffusion_nft':
        with temporary_policy(runtime, runtime['collection_state']):
            return _collect(runtime, case, method, steps, training)
    return _collect(runtime, case, method, steps, training)


def _collect(runtime, case, method, steps, training):
    started = time.monotonic()
    if training and method not in METHODS:
        raise ValueError('Unknown learning method.')
    prepare(runtime)
    device = runtime['device']
    schedule = schedule_for(runtime, steps)
    semantic_schedule = schedule_for(runtime, steps)
    if (not torch.equal(schedule.timesteps, semantic_schedule.timesteps)
            or not torch.equal(schedule.sigmas, semantic_schedule.sigmas)):
        raise ValueError('RGB and semantic scheduler clocks differ.')
    seeds = {name: case[name + '_seed'] for name in ('rgb', 'action', 'semantic')}
    campaign._check_seeds(seeds)
    rgb_generator = torch.Generator(device=device).manual_seed(seeds['rgb'])
    action_generator = torch.Generator(device=device).manual_seed(seeds['action'])
    semantic_generator = torch.Generator().manual_seed(seeds['semantic'])
    initial_rng = {'rgb_before': rgb_generator.get_state(), 'action_before': action_generator.get_state(),
                   'semantic_before': semantic_generator.get_state()}
    state = runtime['pipeline'].prepare_latents(
        1, 16, 480, 832, 81, torch.float32, device, rgb_generator, None)
    if tuple(state.shape) != RGB_SHAPE:
        raise ValueError('Unexpected initial RGB latent shape.')
    semantic_state = torch.randn(SEMANTIC_SHAPE, generator=semantic_generator, dtype=torch.float32).to(device)
    initial_hashes = {'rgb': campaign._tensor_hash(state), 'semantic': campaign._tensor_hash(semantic_state)}
    selection_generator = torch.Generator().manual_seed(case.get('step_seed', seeds['action']))
    flash_index = ACTION_STEPS[int(torch.randint(len(ACTION_STEPS), (), generator=selection_generator))]
    weights = flash_weights(schedule.sigmas)
    actions, ledger, forward_contexts = [], [], {}
    for step_index, timestep in enumerate(schedule.timesteps):
        sigma = float(schedule.sigmas[step_index])
        step_size = float(schedule.sigmas[step_index] - schedule.sigmas[step_index + 1])
        context, semantic_flow = context_at(runtime, state, semantic_state, sigma, timestep, case['prompt'])
        is_action = step_index in ACTION_STEPS
        diagnostics = {}
        flow = context['frozen_flow']
        if is_action:
            flow, diagnostics = velocity(runtime, context, sigma)
        mean, standard_deviation = transition(state, flow, step_size)
        next_state = mean
        noise_hash = None
        if is_action:
            noise = torch.randn(state.shape, device=device, generator=action_generator, dtype=torch.float32)
            next_state = mean + standard_deviation * noise
            noise_hash = campaign._tensor_hash(noise)
            save_action = training and (method == 'flow_grpo' or
                                         (method == 'flash_grpo' and step_index == flash_index))
            if save_action:
                saved = _cpu_context(context)
                saved.update({'sigma': sigma, 'h': step_size, 'step_index': step_index,
                              'std': standard_deviation, 'action': next_state.cpu().clone(),
                              'old_mean': mean.cpu().clone(),
                              'old_log_prob': objectives.normal_log_prob(next_state, mean, standard_deviation).cpu(),
                              'weight': weights[step_index] if method == 'flash_grpo' else 1.0})
                actions.append(saved)
            if training and method in ('awm', 'diffusion_nft'):
                forward_contexts[step_index] = semantic_state.detach().cpu().clone()
        semantic_state = semantic_schedule.step(semantic_flow, timestep, semantic_state, return_dict=False)[0].float()
        state = next_state.float()
        if not bool(torch.isfinite(state).all() and torch.isfinite(semantic_state).all()):
            raise FloatingPointError('Nonfinite historical rollout state.')
        ledger.append({'step_index': step_index, 'sigma': sigma, 'h': step_size,
            'adapted': is_action and runtime['arm'] != 'B',
            'lora_enabled': is_action and runtime['arm'] == 'L', 'stochastic': is_action,
            'noise_sha256': noise_hash, 'base_conditional_captures': 1, 'semantic_advances': 1,
            **diagnostics})
    expected_actions = (4 if method == 'flow_grpo' else 1 if method == 'flash_grpo' else 0) if training else 0
    if len(actions) != expected_actions or len(ledger) != 50:
        raise RuntimeError('Collected transition count differs from the registered schedule.')
    return {'rgb_latent': state.detach().cpu(), 'semantic_latent': semantic_state.detach().cpu(),
        'case': dict(case), 'actions': actions, 'forward_contexts': forward_contexts,
        'rng_states': dict(initial_rng, rgb_after=rgb_generator.get_state(),
            action_after=action_generator.get_state(), semantic_after=semantic_generator.get_state()),
        'statistics': {'method': method, 'learning_method': method, 'training': training,
            'steps': steps, 'guidance_scale': 5.0, 'sampler': 'euler_four_gaussian_actions',
            'arm': runtime['arm'], 'seeds': seeds, 'ledger': ledger,
            'outer_gain': 0.25, 'rms_cap': 0.25, 'action_steps': list(ACTION_STEPS),
            'sigmas': schedule.sigmas.cpu().tolist(), 'timesteps': schedule.timesteps.cpu().tolist(),
            'semantic_steps': len(ledger), 'rgb_latent_sha256': campaign._tensor_hash(state),
            'initial_latent_sha256': initial_hashes, 'parameter_version': runtime['parameter_version'],
            'policy_digest': policy_digest(runtime), 'generation_seconds': time.monotonic() - started,
            'semantic_prediction': 'final_historical_semantic_flow_observer_state',
            'flash_selected_index': flash_index if training and method == 'flash_grpo' else None}}


def _forward_examples(runtime, trajectories, round_index):
    """Fresh endpoints at four positions, conditioned on saved semantic states.

    This artificial RGB noise differs from the rollout RGB state. We recompute
    frozen RGB features and q with the saved historical pre-action semantic
    state, without advancing or training the observer. This conditioning
    distribution shift is an explicit limitation of the matching adaptation.
    """
    for clip_index, trajectory in enumerate(trajectories):
        generator = torch.Generator().manual_seed(620000 + round_index * 1000 + clip_index)
        for step_index in ACTION_STEPS:
            sigma = trajectory['statistics']['sigmas'][step_index]
            timestep = trajectory['statistics']['timesteps'][step_index]
            clean = trajectory['rgb_latent'].to(runtime['device'])
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(runtime['device'])
            noisy = (1 - sigma) * clean + sigma * noise
            semantic_state = trajectory['forward_contexts'][step_index].to(runtime['device'])
            context, _ = context_at(runtime, noisy, semantic_state, sigma, timestep, trajectory['case']['prompt'])
            yield clip_index, step_index, sigma, context, clean, noise - clean


def _optimizer_step(runtime, optimizer, gradient_clip):
    trainable = parameters(runtime)
    optimized = [value for group in optimizer.param_groups for value in group['params']]
    if len(optimized) != len(trainable) or {id(value) for value in optimized} != {id(value) for value in trainable}:
        raise RuntimeError('Optimizer tensors do not match the declared trainable policy.')
    for module in (runtime['pipeline'].transformer, runtime['semantic'], runtime['codec']):
        if any(value.grad is not None for value in module.parameters() if not value.requires_grad):
            raise RuntimeError('A frozen model unexpectedly received a gradient.')
    if any(value.grad is not None and not bool(torch.isfinite(value.grad).all()) for value in trainable):
        raise FloatingPointError('Nonfinite policy gradient; no optimizer step applied.')
    norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(value).all()) for value in trainable):
        raise FloatingPointError('Nonfinite policy parameters after the optimizer step.')
    for state in optimizer.state.values():
        if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in state.values()):
            raise FloatingPointError('Nonfinite optimizer state after the optimizer step.')
    return float(norm)


def technical_probe(runtime, optimizer, trajectory, method, config):
    """Exercise one real-size backward early, then restore the disposable step.

    This tests execution, not the reward signal or future memory requirements.
    It reuses the first learning video, so no extra video or checkpoint is made.
    """
    started = time.monotonic()
    prepare(runtime)
    saved_policy = state_dict(runtime)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all() if runtime['device'].type == 'cuda' else None
    old_version = runtime['parameter_version']
    digest_before = policy_digest(runtime)
    result = {'kind': 'disposable_technical_probe', 'method': method,
              'counts_as_training_update': False, 'geometry_certified': False}
    try:
        optimizer.zero_grad(set_to_none=True)
        if method in ('flow_grpo', 'flash_grpo'):
            action = trajectory['actions'][0]
            context = _device_context(runtime, action)
            sigma = action['sigma']
            current, diagnostics = velocity(runtime, context, sigma, checkpointed=True)
            mean, std = transition(context['state'], current, action['h'])
            if not torch.allclose(mean.detach(), context['old_mean'], rtol=1e-5, atol=1e-5):
                raise RuntimeError('Technical probe found inconsistent unchanged-policy replay.')
            reference_mean, _ = transition(context['state'], context['frozen_flow'], action['h'])
            loss = objectives.grpo_loss(objectives.normal_log_prob(context['action'], mean, std),
                context['old_log_prob'], 1.0, objectives.gaussian_kl(mean, reference_mean, std),
                beta=0.004 if method == 'flow_grpo' else 0.0, weight=action['weight'])
        else:
            example = next(_forward_examples(runtime, [trajectory], round_index=0))
            _, _, sigma, context, clean, target = example
            if method == 'diffusion_nft':
                with torch.no_grad(), temporary_policy(runtime, runtime['collection_state']):
                    old_flow, _ = velocity(runtime, context, sigma)
            current, diagnostics = velocity(runtime, context, sigma, checkpointed=True)
            if method == 'awm':
                loss = objectives.awm_loss(current, target, context['frozen_flow'], 1.0)
            else:
                loss = objectives.nft_loss(current, old_flow, context['frozen_flow'],
                                           context['state'], clean, sigma, 1.0)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite disposable-probe loss.')
        before_flow = current.detach().cpu().clone()
        loss.backward()
        norm = _optimizer_step(runtime, optimizer, config['settings']['gradient_clip'])
        with torch.no_grad():
            after_flow, _ = velocity(runtime, context, sigma)
            movement = float((after_flow.cpu() - before_flow).square().mean().sqrt())
        result.update(loss=float(loss.detach()), gradient_norm=norm,
                      velocity_change_rms=movement, correction=diagnostics,
                      warning=None if movement > 0 else 'No visible flow change in this one probe; report only.')
    finally:
        load_state_dict(runtime, saved_policy)
        optimizer.load_state_dict(saved_optimizer)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(saved_cpu_rng)
        if saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)
        runtime['parameter_version'] = old_version
    if policy_digest(runtime) != digest_before:
        raise RuntimeError('The disposable technical update was not restored.')
    result.update(disposable_update_restored=True, seconds=time.monotonic() - started)
    return result


def _validate_batch(runtime, trajectories, rewards, method):
    if method not in METHODS or runtime['arm'] == 'B':
        raise ValueError('Only a declared trainable method may update.')
    if len(trajectories) != 4 or any(len(group) != 4 for group in trajectories):
        raise ValueError('Expected four groups of four fresh learning videos.')
    if tuple(torch.as_tensor(rewards).shape) != (4, 4):
        raise ValueError('Rewards must match the four by four learning batch.')
    expected_digest = policy_digest(runtime)
    if method == 'diffusion_nft':
        from rgb_geometry_campaign.protocol import tensor_digest
        expected_digest = tensor_digest(runtime['collection_state'])
    identities, action_seeds = set(), set()
    for group in trajectories:
        prompt = group[0]['case']['prompt']
        initial_seeds = (group[0]['case']['rgb_seed'], group[0]['case']['semantic_seed'])
        for trajectory in group:
            if id(trajectory) in identities or trajectory.get('learning_consumed', False):
                raise ValueError('A learning trajectory was duplicated or reused.')
            identities.add(id(trajectory))
            case, statistics = trajectory['case'], trajectory['statistics']
            if case['prompt'] != prompt or (case['rgb_seed'], case['semantic_seed']) != initial_seeds:
                raise ValueError('Historical siblings must share prompt and initial RGB/semantic noise.')
            if case['action_seed'] in action_seeds:
                raise ValueError('Learning siblings require distinct action seeds.')
            action_seeds.add(case['action_seed'])
            if (statistics['method'] != method or not statistics['training']
                    or statistics['arm'] != runtime['arm']
                    or statistics['parameter_version'] != runtime['parameter_version']
                    or statistics['policy_digest'] != expected_digest
                    or statistics['sampler'] != 'euler_four_gaussian_actions'
                    or statistics['steps'] != INFERENCE_STEPS):
                raise ValueError('Stale policy or mismatched collection protocol.')
            actions = trajectory['actions']
            if method == 'flow_grpo' and [action['step_index'] for action in actions] != list(ACTION_STEPS):
                raise ValueError('Flow replay requires all four historical actions.')
            if method == 'flash_grpo' and (len(actions) != 1 or actions[0]['step_index'] not in ACTION_STEPS):
                raise ValueError('Flash replay requires one sampled historical action.')
            if method in ('awm', 'diffusion_nft') and set(trajectory['forward_contexts']) != set(ACTION_STEPS):
                raise ValueError('Matching requires four saved historical semantic states.')
            for action in actions:
                if not math.isclose(action['std'], 0.03 * math.sqrt(action['h']), rel_tol=1e-12):
                    raise ValueError('The saved action noise differs from the historical sampler.')
                expected_weight = flash_weights(torch.tensor(statistics['sigmas']))[action['step_index']]
                if method == 'flow_grpo':
                    expected_weight = 1.0
                if not math.isclose(action['weight'], expected_weight, rel_tol=1e-6):
                    raise ValueError('Replay weights must use the fixed full-schedule normalization.')
                saved_log_prob = objectives.normal_log_prob(action['action'], action['old_mean'], action['std'])
                if not torch.allclose(saved_log_prob, action['old_log_prob'], rtol=1e-5, atol=1e-5):
                    raise RuntimeError('Saved action log probability differs from its behavior mean.')


def update(runtime, optimizer, trajectories, rewards, method, round_index, config):
    """Accumulate one micro-example at a time; never backpropagate through rewards."""
    started = time.monotonic()
    prepare(runtime)
    _validate_batch(runtime, trajectories, rewards, method)
    flattened = [trajectory for group in trajectories for trajectory in group]
    relative = objectives.advantages(rewards).flatten()
    if len(flattened) != 16 or relative.numel() != 16:
        raise ValueError('Expected four groups of four newly generated learning videos.')
    metrics = {'advantages': relative.tolist(), 'optimizer_steps': 0, 'effective_policy_updates': 0,
               'retained_update': False, 'effective_update': False, 'passes': [], 'warnings': []}
    if not bool(relative.ne(0).any()):
        for trajectory in flattened:
            trajectory['learning_consumed'] = True
        metrics.update(status='tied_rewards_skipped', update_seconds=time.monotonic() - started)
        return metrics
    replay_method = method in ('flow_grpo', 'flash_grpo')
    passes = 2 if replay_method else 1
    for pass_index in range(passes):
        optimizer.zero_grad(set_to_none=True)
        before_digest = policy_digest(runtime)
        records = []
        probe = None
        if replay_method:
            terms = [(index, action) for index, trajectory in enumerate(flattened) for action in trajectory['actions']]
            for clip_index, action in terms:
                context = _device_context(runtime, action)
                current, diagnostics = velocity(runtime, context, action['sigma'], checkpointed=True)
                mean, std = transition(context['state'], current, action['h'])
                if pass_index == 0 and not torch.allclose(mean.detach(), context['old_mean'], rtol=1e-5, atol=1e-5):
                    raise RuntimeError('Unchanged-policy replay differs from its collected transition.')
                reference_mean, _ = transition(context['state'], context['frozen_flow'], action['h'])
                log_prob = objectives.normal_log_prob(context['action'], mean, std)
                divergence = objectives.gaussian_kl(mean, reference_mean, std)
                coefficient = 0.004 if method == 'flow_grpo' else 0.0
                loss = objectives.grpo_loss(log_prob, context['old_log_prob'], relative[clip_index],
                    divergence, beta=coefficient, weight=action['weight'])
                ratio = float(torch.exp(log_prob.detach() - context['old_log_prob']).mean())
                if probe is None:
                    probe = (context, action['sigma'], current.detach().cpu().clone())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite GRPO loss.')
                (loss / len(terms)).backward()
                records.append({'clip_index': clip_index, 'step_index': action['step_index'],
                                'loss': float(loss.detach()), 'ratio': ratio, 'reference_kl_mean_coordinate': float(divergence.mean().detach()),
                                **diagnostics})
        else:
            for clip_index, step_index, sigma, context, clean, target in _forward_examples(runtime, flattened, round_index):
                if method == 'diffusion_nft':
                    with torch.no_grad(), temporary_policy(runtime, runtime['collection_state']):
                        old_flow, _ = velocity(runtime, context, sigma)
                current, diagnostics = velocity(runtime, context, sigma, checkpointed=True)
                if method == 'awm':
                    loss = objectives.awm_loss(current, target, context['frozen_flow'], relative[clip_index])
                elif method == 'diffusion_nft':
                    loss = objectives.nft_loss(current, old_flow, context['frozen_flow'],
                        context['state'], clean, sigma, relative[clip_index])
                else:
                    raise ValueError('Unknown RL algorithm.')
                if probe is None:
                    probe = (context, sigma, current.detach().cpu().clone())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite forward-matching loss.')
                (loss / 64).backward()
                records.append({'clip_index': clip_index, 'step_index': step_index, 'sigma': sigma,
                                'loss': float(loss.detach()), **diagnostics})
        gradient_norm = _optimizer_step(runtime, optimizer, config['settings']['gradient_clip'])
        metrics['optimizer_steps'] += 1
        after_digest = policy_digest(runtime)
        with torch.no_grad():
            after_flow, _ = velocity(runtime, probe[0], probe[1])
            difference = after_flow.cpu() - probe[2]
            movement = float(difference.square().mean().sqrt())
        effective = before_digest != after_digest and movement > 0
        metrics['effective_policy_updates'] += int(effective)
        if not effective:
            metrics['warnings'].append({'kind': 'no_effective_velocity_change', 'pass_index': pass_index})
        saturated = sum(record.get('clip_fraction', 0) > 0 for record in records)
        if saturated:
            metrics['warnings'].append({'kind': 'residual_cap_active', 'pass_index': pass_index,
                                        'terms': saturated, 'report_only': True})
        outside_clip = sum(not 0.999 <= record['ratio'] <= 1.001 for record in records if 'ratio' in record)
        if outside_clip:
            metrics['warnings'].append({'kind': 'likelihood_ratio_outside_clip', 'pass_index': pass_index,
                                        'terms': outside_clip, 'report_only': True})
        metrics['passes'].append({'pass_index': pass_index, 'loss_mean': sum(row['loss'] for row in records) / len(records),
                                 'gradient_norm': gradient_norm, 'velocity_change_rms': movement,
                                 'parameter_digest_before': before_digest, 'parameter_digest_after': after_digest,
                                 'terms': records})
        runtime['parameter_version'] += 1
        del probe
    if method == 'diffusion_nft':
        decay = min(0.001 * round_index, 0.5)
        current_values = state_dict(runtime)
        runtime['collection_state'] = {name: decay * runtime['collection_state'][name] + (1 - decay) * value
                                       for name, value in current_values.items()}
        metrics['collection_ema_decay'] = decay
    optimizer.zero_grad(set_to_none=True)
    for trajectory in flattened:
        trajectory['learning_consumed'] = True
    metrics.update(status='updated', retained_update=True,
                   effective_update=metrics['effective_policy_updates'] > 0,
                   update_seconds=time.monotonic() - started)
    return metrics
