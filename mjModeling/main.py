import multiprocessing as mp
from Oscillator import run_drawer
from mjModeling.conf import ROBOT_SCENE, workingPiece
from kuka_iiwa_14.iiwa14_model import iiwa14
from mjModeling.controllers import (JacobianIK, BasicVariableImpedanceControl,
                                    GMRVariableImpedanceControl,
                                    PassiveVariableImpedanceControl,
                                    OptimizedVariableImpedanceControl, VariableImpedanceControl)
from mjModeling.cutting_materials import Material
from mjModeling.experiments.motion import InitPos, straightCutting
from mjModeling.experiments import Experiment
from visualization.visualizer import Visualize
from data import MaterialData
# 1 - build experiment env
robot = iiwa14().create(ROBOT_SCENE)
working_piece = Material()

bp_data = MaterialData.cork
working_piece.cut_resistance = workingPiece.MATERIAL_RESISTANCE.value
working_piece.surface_hight = 0.04
# Experiments
straight_cut: Experiment = straightCutting(robot)
init_pos: Experiment = InitPos(robot)

# Controllers
vic = VariableImpedanceControl(robot, gmr_sequence=cork_data)
vic.working_piece = working_piece

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
