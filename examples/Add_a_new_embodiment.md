## Add a new embodiment 

when we need to train on a new embodiment (or a new dataset, i mean, the action/state format or task dynamtics are changed even if the embodiment is the same)

1. add a new embodiment tag

    ```
        class EmbodimentTag(Enum):
            ...
            PSIX_HE_G1 = "psix_he_g1"
    ```

2. add a new path pointing to the new datasetset at the command line arguments

    ```
        --data.dataset_paths
        ...
        psix_he_g1:/hfm/data/psix_he_anno_0512/manual_label_v1_0512_lerobot/g1
    ```

3. add a corresponding `field transform` in 
    ```
        class MixedFieldTransform(FieldTransform):
            ...
            psix_he_g1: PsixHEFieldTransform

            def get_all_transforms(self) -> dict[str, FieldTransform]:
                return {
                    ...
                    "psix_he_g1": self.pisx_he_g1.get_transform()
                }
    ```

4. define PsixHEFieldTransform, checkout the comments in the code

    (Try not to have inheritance between Transform classess as it will complicate things, code redundancy is encourage here)

    ```
        class PsixHEFieldTransform(FieldTransform):
            ##### key mappings for video/action/state/instruction 
            ##### refer to /path/to/lerobot/meta/modality.json
            
            video_apply_to: list[str] = Field(
                default_factory=lambda: [
                    "video.egocentric"
                    # ... where multi-view cam is configured
                ]
            )

            state_apply_to: list[str] = Field(
                default_factory=lambda: [
                    # can stay at high level, eg.,
                    "state.joint_state",
                    "state.lower_body_cmd"
                ]
            )

            state_apply_to: list[str] = Field(
                default_factory=lambda: [
                    # prefer detailed annotations, 
                    # because it is used in l1 loss training log

                    # first-order joint positions
                    "state.left_hand",
                    "state.right_hand",
                    "state.left_arm",
                    "state.right_arm",

                    # Optional torso and lower-body joints
                    "state.left_leg",
                    "state.right_leg",
                    "state.torso",

                    # Optional eef positions
                    "state.left_eef",
                    "state.right_eef",

                    # Optional high-level lower body states
                    "state.base_vx",
                    "state.base_vy",
                    "state.base_vyaw",
                    "state.base_target_yaw",
                    "state.base_height",
                ]
            )
            ##### key mapping ended #####

            ### video transforms ###
            video_to_tensor: ...
            video_crop: ...
            video_resize: ...
            video_color_jitter: ... 
            video_to_numpy: ...


            ### State transforms ###
            state_to_tensor: ...
            # where stat normalization happens
            state_transform: ... 

            ### Action transforms ###
            action_to_tensor: ...
            action_transform: ...

            #!! Concat the modality keymapping as a single field ###
            concat: ConcatTransform

            .... copy other boilerplate code too 

    ```

5. manually create a `modality.json` in `/path/to/lerobo/meta`
    ```
        {
            "state": {
            },
            "action": {
            },
            "video": {
            },
            "annotation": {
            }
        }
    ```