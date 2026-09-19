# EmblaV1

## Extending DreamerV3 for robotic application and better Morphology awareness by implementing a FFKSM

# EmblaV2

## Using EmblaV1 with a biologically inspired intrinsic reward system

## Details

A detailed description of EmblaV1 is provided at additionalMaterials/DreamerFFKSM.pdf.

## Usage

* install requirements (note that you need torch with cuda)
* replace pusher_v5.xml or ant_v5.xml in the gymnasium lib folder (yourPythonVenv/gymnasium/envs/mujoco/assets) with the file in this repo! (necessary for the colourfilter)
* run main.py to start
* modify config to change all relevant Dreamer and FFKSM parameters

## Current state

* EmblaV1 done -> Result: DreamerV3 is already very good with limited vision, sm is redundant for EmblaV1
* EmblaV2 has self-sustainability as a first intrinsic reward
* needs testing

## Acknowledgements

* [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer) Helped me to understand the DreamerV3 Architecture. Used the Code as a base.
* [SelfModel](https://github.com/H-Y-H-Y-H/SelfSimRobot) Original Selfmodel-code, which I modified and fused with the DreamerV3 Architecture.

