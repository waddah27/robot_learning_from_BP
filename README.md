# Overview
Robot learning from human behaviour priors (BP) is an approach derived from robot learning from human demonstration and skill transfer learning methodologies
where robots learn skills combining motion planning and contact-rich for doing tasks where both kinematics and dynamics are critical. This learning approach involves scenarios
of teaching the robot human kinetic and dynamic skills offline where the human is out of the loop. i.e, encoding skills as Behaviour priors via probabilistic models that are trained
on real records and act as Behaviour prior generator.

The project is on process and the details will be added later.

Just for record, the project uses pyqt6 to create oscillator, To make it work u should install on of the platform's plugins:
Available platform plugins are: `vnc, eglfs, wayland-brcm, wayland-egl, wayland, xcb, vkkhrdisplay, offscreen, minimal, linuxfb, minimalegl`.

I chose to install following plugin
```bash
sudo apt-get update
sudo apt-get install libxcb-cursor0

```
This setup depends on scipy to calculate the planned continuous motion from gmr, if `scipy` wwas not successfully installed via
`pip install -r requirements` then consider installing it to system site packages as follows:

```bash
sudo apt install python3-scipy
```

note that this setup runs cutting experiment in different scenarios to prove multiple contributions of this research:
1. Doing straigt cuts from learnt behaviour priors which are generated from bp model and are given as sequences of way-points during time instances where the cutting material (work piece) is static.
2. Same above way-point approach but with mobile work piece.
3. Straight cuts from learned behaviour priors generated from bp model then parametrized using phase variable to provide continuous trajectory tracking and scalable time duration of task execution where the work piece is static.
4. same above continuous trajectory approach with mobile work piece.

selection which scenario to observe is very easy .. go to `mjModeling/conf/configs.py` and modify `CONTINUOUS_TRAJ` in `paramVIC` and `MOBILE` in `workPiece` ..

To test phase variable effect on task parametrisation (run faster or slower) simply adjust `paramVIC.PHASE_SPEED`.

Task timing and its validation disturbance can be switched independently in
`mjModeling/conf/configs.py`:

```python
paramVIC.STATE_DRIVEN_PHASE = True   # False: matched time-driven phase
paramVIC.HOLDBACK_DISTURBANCE = False
paramVIC.HOLDBACK_FORCE_N = 120.0
paramVIC.HOLDBACK_START_S = 0.9
paramVIC.HOLDBACK_END_S = 1.4
```

Retrain the material GMR priors from progress-registered demonstrations with:

```bash
python training/train_gmr_priors.py
python variability_control/variance_gains.py
```

The first command replaces the legacy notebook workflow, which aligned trials
by sample index. It registers each trial by monotone spatial cutting progress,
fits standardized material-specific GMMs, and writes the runtime `.npy` priors.

The holdback is disabled for nominal execution. When enabled, it applies a
bounded external force opposite the local cutting-path tangent; it is intended
only for controlled phase-synchronization validation.
you can also play with other configurations but not all cases and configs are guaranteed to give appropriate behaviour as my research is not about choosing the best vic controller but how to use skill patterns learnt from human to teach a robot
```python
class paramVIC:
    VIC_MAX_STEPS = 5000 #configs["vic_params"]["vic_max_steps"]
    VIC_TOL = 0.004 #configs["vic_params"]["tolerance"]
    VIC_KP_MIN = 400.0 #configs["vic_params"]["kp"]["min"]
    VIC_KP_MAX = 1500.0 #configs["vic_params"]["kp"]["max"]
    VIC_M = 1 #configs["vic_params"]["m"]
    VIC_KI = 200 #configs["vic_params"]["ki"]
    VIC_LAMBDA_SQ = 1e-4 #configs["vic_params"]["lambda_sq"]
    ADAPTIVE = True
    CONTINUOUS_TRAJ = True
    PHASE_SPEED = 1.0

# GMR Integration Parameters
class GMRParams:
    FORCE_ERROR_THRESHOLD = 5.0  # N, threshold for triggering adaptation
    ADAPTATION_RATE = 0.1  # How quickly to adapt gains
    BLENDING_FACTOR = 0.7  # Weight of GMR vs reactive control (0=GMR only, 1=reactive only)
    MIN_STIFFNESS_CONTACT = 100.0  # N/m, minimum in contact
    MAX_STIFFNESS_FREE = 2000.0  # N/m, maximum in free motion

# Add to existing paramVIC class
paramVIC.GMR = GMRParams

class workPiece:
    MATERIAL_RESISTANCE = configs["material_params"]["material_resistance"]
    MATERIAL_NAME = configs["material_params"]["material_name"]
    MATERIAL_IS_SOLID = True
    POS = np.array([0.5, 0.0, 0.02])
    SIZE = np.array([0.3, 0.3, 0.02])
    SURFACE: float = SIZE[2] + POS[2]
    MOBILE: bool = False

```
