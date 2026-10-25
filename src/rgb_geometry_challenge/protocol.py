"""Fixed 40-prompt challenge panel. Planning uses no model or GPU imports."""

import copy

from rgb_geometry_qualitative import protocol as heads
from rgb_geometry_qualitative_lora import protocol as lora


ROOT = heads.ROOT
CONFIG = ROOT / "configs/rgb_geometry_challenge_v1.json"
METHODS = ("rloo", "flow_grpo", "flash_grpo")
ARMS = ("R", "S", "L")
VARIANTS = ("baseline",) + tuple(method + "_" + arm for method in METHODS for arm in ARMS)
EXPERIMENTS = VARIANTS
CLASS_NAMES = heads.CLASS_NAMES
MINIMUM_FREE_BYTES = 10_000_000_000
read_json = heads.read_json
write_json = heads.write_json
file_hash = heads.file_hash
fingerprint = heads.fingerprint
append_row = heads.append_row
runtime_args = heads.runtime_args
tracker_probe = heads.tracker_probe
safe_path = heads.comparison.campaign.previous.safe_path

COMMON_SCENE = (
    "Realistic daytime street scene with a paved road, a sidewalk, low-rise buildings, "
    "green trees and an open sky. Clear daylight. One continuous five-second shot "
    "without a cut. The main action stays within view. "
)
FIXED_CAMERA = (
    "A stationary camera at pavement height shows the whole action in a moderately "
    "wide view. The main objects are close enough to see their outlines and parts. "
)
BACKGROUND_CLASSES = ["road", "sidewalk", "building", "vegetation", "sky"]


def scene(number, category, action, objects, *, camera="fixed", family=None,
          condition="original", requested_counts=None, review_focus=""):
    """Keep the common setting identical while recording the intended challenge."""
    camera_text = FIXED_CAMERA if camera == "fixed" else ""
    return {
        "scene": f"challenge_{number:02d}",
        "prompt": COMMON_SCENE + camera_text + action,
        "expected_classes": objects + BACKGROUND_CLASSES,
        "category": category,
        "family": family,
        "condition": condition,
        "camera": camera,
        "requested_counts": requested_counts or {},
        "review_focus": review_focus,
        "prompt_adherence": "unverified_until_video_review",
    }


def prompts():
    """Twenty-five initial scenes, followed by 15 density/speed variations."""
    rows = [
        scene(0, "vehicle_shape", "A single red sedan drives slowly from left to right "
              "along the road in side view, staying fully visible throughout the shot.",
              ["car"], family="car_pass", condition="control", requested_counts={"car": 1},
              review_focus="Body, roof, windows and wheels during translation."),
        scene(1, "vehicle_shape", "A red sedan makes a gentle left turn across the view. "
              "Its front is visible first, followed by its side; the complete car stays in frame.",
              ["car"], review_focus="Vehicle parts and plausible perspective during turning."),
        scene(2, "vehicle_shape", "A blue hatchback follows a gentle curve towards the camera. "
              "The whole car remains visible as its front and side become larger.",
              ["car"], review_focus="Body shape and scale changes as a car approaches."),
        scene(3, "vehicle_shape", "A single yellow city bus slowly turns around a broad street "
              "corner. Its front and long side are visible and its whole body remains in frame.",
              ["bus"], family="bus_turn", condition="control", requested_counts={"bus": 1},
              review_focus="Long body, windows and wheels during a turn."),
        scene(4, "vehicle_shape", "A single white box truck slowly turns around a broad street "
              "corner. Its cab and cargo box remain fully visible throughout the shot.",
              ["truck"], family="truck_turn", condition="control", requested_counts={"truck": 1},
              review_focus="Connection between cab, cargo box and wheels."),
        scene(5, "overlap_and_separation", "A red car and a blue car move slowly along separate "
              "lanes with a clear visible gap between them throughout the shot.",
              ["car"], requested_counts={"car": 2}, review_focus="Simple two-object control."),
        scene(6, "overlap_and_separation", "A red car drives slowly in front of a parked blue "
              "car. The blue car is fully visible at first, partly hidden in the middle of the "
              "shot, and visible again before the shot ends.", ["car"],
              requested_counts={"car": 2}, review_focus="Temporary overlap and separation of two cars."),
        scene(7, "overlap_and_separation", "A red car and a blue car travel in opposite "
              "directions in separate lanes. Their outlines briefly overlap in the camera "
              "view and separate again before the shot ends.", ["car"],
              requested_counts={"car": 2}, review_focus="Opposite motion, image overlap and object identity."),
        scene(8, "overlap_and_separation", "A white bus drives slowly past a parked blue bus. "
              "Both buses are visible before and after the moving bus partly obscures the "
              "parked bus in the middle of the shot.", ["bus"], requested_counts={"bus": 2},
              review_focus="Separate bus bodies and windows during overlap."),
        scene(9, "overlap_and_separation", "A red car and a blue car drive side by side in "
              "separate lanes. The red car slowly moves ahead, while both whole cars remain "
              "in the camera view.", ["car"], requested_counts={"car": 2},
              review_focus="Distinct vehicle bodies while relative positions change."),
        scene(10, "occlusion", "A single pedestrian wearing a blue jacket walks from left "
              "to right along the sidewalk. Their full body remains visible throughout the shot.",
              ["person"], family="pedestrian_crossing", condition="control",
              requested_counts={"person": 1}, review_focus="Full-body walking control."),
        scene(11, "occlusion", "A pedestrian in a blue jacket walks along the sidewalk "
              "behind a parked car. Their full body is visible at first, their lower body is "
              "briefly hidden by the car, and their full body is visible again before the end.",
              ["person", "car"], review_focus="Pedestrian shape and reappearance after partial occlusion."),
        scene(12, "occlusion", "A red car drives behind a parked bus. The car is visible "
              "before reaching the bus, hidden briefly, then fully visible again before the "
              "five-second shot ends.", ["car", "bus"],
              review_focus="Vehicle identity before and after complete occlusion."),
        scene(13, "occlusion", "A cyclist rides slowly from left to right behind one narrow "
              "tree trunk. The cyclist and bicycle are visible before and after the brief "
              "partial occlusion, with both bicycle wheels inside the image.",
              ["person", "bicycle"], review_focus="Thin bicycle parts through partial occlusion."),
        scene(14, "occlusion", "A blue car drives slowly past a narrow tree trunk. Only a "
              "small part of the car is hidden at a time; the whole car is visible both "
              "before and after it passes the tree.", ["car"],
              review_focus="Window and body continuity behind a narrow occluder."),
        scene(15, "pedestrian_and_bicycle", "One person stands beside a stationary bicycle "
              "on the sidewalk. The full person, bicycle frame and both wheels remain visible.",
              ["person", "bicycle"], review_focus="Static fine-structure control."),
        scene(16, "pedestrian_and_bicycle", "A pedestrian takes several normal walking steps "
              "towards the camera along the sidewalk, remaining fully visible from head to feet.",
              ["person"], review_focus="Body proportions during approach and walking."),
        scene(17, "pedestrian_and_bicycle", "A single cyclist rides slowly from left to right "
              "along a clear bicycle lane. The rider, bicycle frame and both wheels remain "
              "visible in side view throughout the shot.", ["person", "bicycle"],
              family="cyclist_pass", condition="control",
              requested_counts={"person": 1, "bicycle": 1},
              review_focus="Rider limbs, frame and wheels in side view."),
        scene(18, "pedestrian_and_bicycle", "A cyclist follows a gentle curve so the bicycle "
              "changes from side view to a three-quarter view. The rider and both wheels "
              "stay inside the image.", ["person", "bicycle"],
              review_focus="Plausible bicycle structure during viewpoint changes."),
        scene(19, "pedestrian_and_bicycle", "A pedestrian walks from left to right beside "
              "a bicycle, pushing it along the sidewalk. Both the full person and bicycle "
              "remain visible, including their hand on the handlebar.", ["person", "bicycle"],
              review_focus="Contact between hand and handlebar and separation of legs and frame."),
        scene(20, "camera_and_scene", "The stationary camera looks along a straight street "
              "with a clear pavement edge and a rectangular building corner. No large object "
              "passes in front of these edges.", [], review_focus="Static road and building structure."),
        scene(21, "camera_and_scene", "The camera moves slowly sideways past one parked red "
              "car, gradually revealing its front and side. The complete car remains in "
              "view, with the pavement and a building behind it.", ["car"], camera="sideways",
              review_focus="Car geometry and background parallax during lateral camera motion."),
        scene(22, "camera_and_scene", "The camera moves slowly forward beside a straight "
              "pavement edge and a row of parked cars. Near and distant cars and the building "
              "edges remain visible.", ["car"], camera="forward",
              review_focus="Road boundaries, parallax and vehicle structure during camera motion."),
        scene(23, "camera_and_scene", "The camera moves slowly sideways past a rectangular "
              "building corner, revealing the adjoining wall and the pavement below. Both "
              "walls and the pavement remain in view.", [], camera="sideways",
              review_focus="Wall junctions, corner shape and pavement relationships."),
        scene(24, "camera_and_scene", "The camera moves slowly forward along a gently "
              "curving street towards a parked bus. The full bus, the curved pavement edge "
              "and nearby buildings remain in view.", ["bus"], camera="forward",
              review_focus="Road curvature, bus structure and perspective during approach."),
    ]

    # These five families change density, speed, or both while keeping the camera fixed.
    crowded = [
        scene(25, "crowded", "Six cars of different colours drive slowly from left to "
              "right along three lanes. Several cars partly overlap in the view and then "
              "separate, with gaps between the lanes. The nearest cars remain large enough "
              "to inspect their windows and wheels.", ["car"], family="car_pass",
              condition="crowded", requested_counts={"car": 6},
              review_focus="Distinct car bodies, count and identity in a crowded view."),
        scene(26, "crowded", "Three city buses, yellow, white and blue, slowly turn around "
              "a broad street corner in separate lanes. Their long sides partly overlap "
              "in the view and separate again. Each bus is visible before and after the overlap.",
              ["bus"], family="bus_turn", condition="crowded", requested_counts={"bus": 3},
              review_focus="Long vehicle bodies, windows and separation during turns."),
        scene(27, "crowded", "Three box trucks, white, blue and red, slowly turn around a "
              "broad street corner in separate lanes. Their cabs and cargo boxes partly "
              "overlap in the view, then separate again before the end of the shot.",
              ["truck"], family="truck_turn", condition="crowded", requested_counts={"truck": 3},
              review_focus="Separate cab-box pairs and wheels through overlap."),
        scene(28, "crowded", "Eight pedestrians in different coloured jackets walk along "
              "a broad sidewalk in both directions. People pass in front of one another, "
              "briefly hiding parts of their bodies, then separate. Several full bodies "
              "remain close enough to inspect.", ["person"], family="pedestrian_crossing",
              condition="crowded", requested_counts={"person": 8},
              review_focus="Separate people, body parts and reappearance after occlusion."),
        scene(29, "crowded", "Six cyclists ride slowly from left to right along a broad "
              "bicycle lane in two staggered rows. Riders and bicycles briefly overlap in "
              "the view, then separate. The nearest bicycle frames and wheels remain visible.",
              ["person", "bicycle"], family="cyclist_pass", condition="crowded",
              requested_counts={"person": 6, "bicycle": 6},
              review_focus="Distinct riders and bicycle frames in a group."),
    ]
    faster = [
        scene(30, "faster", "A single red sedan drives briskly from left to right along "
              "the road in side view. It moves several car lengths during the shot while "
              "remaining fully visible; its body, windows and wheels can still be seen.",
              ["car"], family="car_pass", condition="faster", requested_counts={"car": 1},
              review_focus="Car geometry under faster translation; distinguish blur from deformation."),
        scene(31, "faster", "A single yellow city bus turns briskly around a broad street "
              "corner, completing the visible change from front to side view during the "
              "shot. The whole bus remains in frame, with its windows and wheels visible.",
              ["bus"], family="bus_turn", condition="faster", requested_counts={"bus": 1},
              review_focus="Long body and wheel layout under faster turning."),
        scene(32, "faster", "A single white box truck turns briskly around a broad street "
              "corner. It makes a clear change from front to side view during the shot, "
              "with its entire cab and cargo box still in frame.", ["truck"], family="truck_turn",
              condition="faster", requested_counts={"truck": 1},
              review_focus="Cab-box relationship and wheel layout under faster turning."),
        scene(33, "faster", "A single pedestrian wearing a blue jacket jogs briskly from "
              "left to right along the sidewalk. Their full body remains in view while "
              "their arms and legs move through several running strides.", ["person"],
              family="pedestrian_crossing", condition="faster", requested_counts={"person": 1},
              review_focus="Body proportions and limbs during faster movement."),
        scene(34, "faster", "A single cyclist rides briskly from left to right along a "
              "clear bicycle lane in side view. The rider pedals quickly and moves several "
              "bicycle lengths while the complete rider and both wheels remain in frame.",
              ["person", "bicycle"], family="cyclist_pass", condition="faster",
              requested_counts={"person": 1, "bicycle": 1},
              review_focus="Rider-frame relationships and wheels under faster movement."),
    ]
    combined = [
        scene(35, "crowded_and_faster", "Six cars of different colours drive briskly from "
              "left to right along three lanes. Some move ahead of others; their outlines "
              "briefly overlap in the view and separate again. The nearest cars stay in "
              "frame long enough to see their windows and wheels before and after overlap.",
              ["car"], family="car_pass", condition="crowded_and_faster", requested_counts={"car": 6},
              review_focus="Car merging, shape and identity with both crowding and faster motion."),
        scene(36, "crowded_and_faster", "Three city buses, yellow, white and blue, turn "
              "briskly around a broad street corner in separate lanes. Their long sides "
              "briefly overlap in the view then separate, with each bus visible before "
              "and after the overlap within this shot.", ["bus"], family="bus_turn",
              condition="crowded_and_faster", requested_counts={"bus": 3},
              review_focus="Distinct bus bodies and windows under crowded faster turning."),
        scene(37, "crowded_and_faster", "Three box trucks, white, blue and red, turn "
              "briskly around a broad street corner in separate lanes. Their cabs and "
              "cargo boxes briefly overlap in the view and then separate, all within "
              "the shot.", ["truck"], family="truck_turn", condition="crowded_and_faster",
              requested_counts={"truck": 3},
              review_focus="Distinct trucks, cab-box links and reappearance after overlap."),
        scene(38, "crowded_and_faster", "Eight pedestrians in different coloured jackets "
              "jog along a broad sidewalk in both directions. They pass in front of one "
              "another, briefly hiding body parts, then separate. Several runners remain "
              "fully in frame long enough to inspect their arms and legs.", ["person"],
              family="pedestrian_crossing", condition="crowded_and_faster",
              requested_counts={"person": 8},
              review_focus="Body merging, limb changes and identity in a moving crowd."),
        scene(39, "crowded_and_faster", "Six cyclists ride briskly from left to right "
              "along a broad bicycle lane in two staggered rows. Some riders overtake "
              "others, briefly overlapping in the view then separating. The nearest "
              "riders and bicycles remain in frame before and after the overlap.",
              ["person", "bicycle"], family="cyclist_pass", condition="crowded_and_faster",
              requested_counts={"person": 6, "bicycle": 6},
              review_focus="Rider and bicycle merging, wheels and reappearance at greater speed."),
    ]
    return rows + crowded + faster + combined


def expected_config():
    original = heads.expected_config()
    settings = copy.deepcopy(original["settings"])
    settings["rounds"] = 0
    settings["retention"] = "final_rgb_mp4_and_json_csv_no_new_checkpoints"
    provenance = copy.deepcopy(original["provenance"])
    provenance["evaluation_panel_previously_inspected"] = False
    provenance["historical_panel_previously_inspected"] = True
    provenance["note"] = (
        "New prompts chosen after reviewing the earlier qualitative study. "
        "The new generated outputs have not yet been inspected. This is an exploratory "
        "challenge study using the existing round-10 policies, not a blinded confirmation "
        "test. Visible failures and tracking scores do not certify 3D geometry."
    )
    return {
        "format": "rgb_geometry_challenge_v1",
        "inference_only": True,
        "settings": settings,
        "experiments": list(VARIANTS),
        "method_settings": copy.deepcopy(lora.expected_config()["method_settings"]),
        "checkpoint_training_reference": {
            "completed_round": 10,
            "training_seed": original["settings"]["seed"],
            "note": "Historical training metadata only; this campaign performs no updates.",
        },
        "source_recipes": copy.deepcopy(original["source_recipes"]),
        "provenance": provenance,
        "panels": {"evaluation": prompts()},
        "sampling": {
            "seeds_per_prompt": 2,
            "rgb_seed_base": 610001,
            "semantic_seed_offset": 100000,
            "action_seed_offset": 200000,
            "same_case_noise_across_variants": True,
            "generation_seed_is_not_training_seed": True,
        },
        "retention": {
            "videos_per_variant": 80,
            "total_videos": 800,
            "video_format": "RGB_H264_MP4",
            "save_raw_reward_points": True,
            "save_new_checkpoints": False,
            "copy_frozen_models": False,
        },
        "interpretation": {
            "study": "exploratory_inference_challenge_with_existing_round_10_weights",
            "difficulty": "intended_not_empirically_validated",
            "prompt_adherence": "manually_check_actual_density_motion_and_occlusion",
            "geometry_certified": False,
            "independent_human_evaluation": False,
            "review_without_reward_first": True,
            "proxy_limit": "Class persistence cannot establish 3D geometry accuracy.",
            "semantic_limit": "The frozen codec maps person and bicycle into Other.",
            "comparison_limit": "Original method learning rates and update counts differ.",
            "family_limit": "Related prompts keep the camera fixed, but are not identical scenes or a calibrated speed experiment.",
        },
    }


def load_config(path):
    config = read_json(safe_path(path, existing=True))
    if config != expected_config():
        raise ValueError("The fixed 40-prompt challenge configuration changed.")
    return config


def cases(config, panel="evaluation"):
    if panel != "evaluation":
        raise ValueError("The inference challenge has only an evaluation panel.")
    settings = config["sampling"]
    rows = []
    for scene_index, prompt in enumerate(config["panels"][panel]):
        for repeat in range(settings["seeds_per_prompt"]):
            seed = settings["rgb_seed_base"] + scene_index * 2 + repeat
            rows.append(dict(
                prompt, case_id=f"challenge_{scene_index:02d}_{repeat}", role=panel,
                seed=seed, rgb_seed=seed,
                semantic_seed=seed + settings["semantic_seed_offset"],
                action_seed=seed + settings["action_seed_offset"],
            ))
    return rows


def experiment_spec(experiment):
    """Return the unchanged historical training specification for safe restoration."""
    if experiment not in VARIANTS:
        raise ValueError("Unknown challenge variant: " + str(experiment))
    if experiment.endswith("_L"):
        return lora.experiment_spec(experiment)
    return heads.experiment_spec(experiment)
