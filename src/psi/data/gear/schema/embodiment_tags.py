from enum import Enum


class EmbodimentTag(Enum):
    """
    PSI0 format decoupled whole-body loco-manipulation data (36D action, 32D state, egocentric).
    """
    PSI0_AMO_G1 = "psi0_amo_g1"

    """
    PSI0 SONIC format wholebody loco-manipulation data (64 latent action + 7 * 2 DoF hands).
    """
    PSI0_SONIC_G1 = "psi0_sonic_g1"

    """
    SIMPLE G1 humanoid with Psi0 decouple wbc loco-manipulation data (36D action, 32D state, egocentric).
    """
    SIMPLE_G1 = "simple_g1"


    """
    DECOUPLED WBC G1 humanoid with pi07 style annotations (sub-goal image, sub-task prompt, metadata etc).
    Used by the PsiX subtask pack (manual_label_v1_0512_lerobot/g1/).
    """
    DECOUPLED_WBC_G1 = "decoupled_wbc_g1"

    """
    DECOUPLED WBC H1 humanoid with pi07 style annotations (sub-goal image, sub-task prompt, metadata etc).
    Used by the PsiX subtask pack (manual_label_v1_0512_lerobot/h1/). Parallel to DECOUPLED_WBC_G1.
    """
    DECOUPLED_WBC_H1 = "decoupled_wbc_h1"

    """
    PSIX_HE G1 + sonic body tokens: action.body_token (64D latent) produced by data_sonic.json.
    Includes all subtask/memory annotations from manual_label_v1_0512_psix.
    """
    PSIX_HE_G1_SONIC = "psix_he_g1_sonic"

    """
    PSIX G1 + sonic with the NECK DoF (e.g. g1_neck_0617 / diverse_tasks): a distinct embodiment,
    NOT an fps alias. ONE RGB video (observation.images.egocentric 672x384), 45D state
    ([hand|arm|leg|neck2]) and an action.neck(2) column -> action path 14+64+2 = 80D. 30fps / wide
    cam, so it needs its own metadata-merge bucket. Old 78D/43D packs pad up to 80D/45D in mixing.
    """
    PSIX_G1_SONIC_NECK = "psix_g1_sonic_neck"

    """
    PSIX_HE H1 + sonic body tokens: parallel to PSIX_HE_G1_SONIC for H1 embodiment.
    """
    PSIX_HE_H1_SONIC = "psix_he_h1_sonic"

    """
    EgoDex full-corpus pretrain packs (raw_egodex_to_psix.py): 480x270 egocentric video, 48D
    world-frame absolute pose + per-frame 16D camera_extrinsic, no_annotation.
    """
    EGODEX_PSIX = "egodex_psix"
