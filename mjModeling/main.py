import multiprocessing as mp
from Oscillator import run_drawer
from mjModeling.conf import ROBOT_SCENE
from kuka_iiwa_14.iiwa14_model import iiwa14
from mjModeling.controllers import JacobianIK, VariableImpedanceControl
from mjModeling.cutting_materials import Material
from mjModeling.experiments.motion import InitPos, straightCutting
from mjModeling.experiments import Experiment
from visualization.visualizer import Visualize
import sys
# 1 - build experiment env
work_piece = Material().from_work_piece()
work_piece.bp_data = 'cork'
robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=work_piece)

# Experiments
straight_cut: Experiment = straightCutting(robot)
init_pos: Experiment = InitPos(robot)

# Controllers
vic = VariableImpedanceControl(robot, use_behaviour_priors=True)
vic.working_piece = work_piece

jik = JacobianIK(robot)
print(f"Gravity = {robot.model.opt.gravity}")
# 2 - simulator
visualizer = Visualize(robot)

if __name__ == '__main__':
    #  Start Oscillator Process
    drawer_proc = mp.Process(target=run_drawer, args=(robot.shm.name,))
    drawer_proc.start()
    controllers = {
        "vic": vic,
        "jik": jik
    }

    experiments = {
        "init_pos": init_pos,
        "straight_cut": straight_cut
        }
    current_experiment = experiments.get("straight_cut")

    current_experiment.controller = vic
    visualizer.run(callback=lambda x: current_experiment.execute(x))
