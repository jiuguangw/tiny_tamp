# Tiny Task and Motion Planning

A minimal task and motion planning package for quick prototyping

## Install

`python -m pip install -e .`

## Use

`python tamp.py --vis-sim`


## Debugging

To debug, you can visualize the belief in the digital twin simulator with
`python tamp.py --vis-belief`

By adding the following, you can pause the planning and view the current state of the simulator in pybullet.
Pressing enter in the terminal will resume planning.
`pbu.wait_if_gui(client=...)`


## TODO

- [ ] create statistics upon failure
- [ ] test the antipodal grasp sampler
- [ ] ycb objects
- [ ] test on real robot