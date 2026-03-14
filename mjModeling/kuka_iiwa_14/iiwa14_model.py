from logger import Logger
from multiprocessing import shared_memory
import os
from typing import Callable

from mjModeling.cutting_materials import Material
import numpy as np
from mjModeling import Robot
from mjModeling.conf import (
    MATERIAL_GEOM,
    SCALPEL_GEOM,
    SCALPEL_HANDLER_1_PATH,
    SCALPEL_HANDLER_2_PATH,
    SCALPEL_PATH,
    FORCE_HISTORY,
)
import mujoco

from mjModeling.conf.configs import oscillatorConfigs as oscConf
from shmemory import SharedMemoryBuffer

__all__ = ["iiwa14"]


class iiwa14(Robot):
    def __init__(self):
        self._model = None
        self._data = None
        self.robot_state = {}
        self.work_piece: Material = None
        # for iiwa14 there are 7 DOFs
        self.nq_robot = 7
        self.buffer: SharedMemoryBuffer = None
        self.reset_state()

    def set_shm_buffer(self):
        self.robot_state["shared_array"] = self.buffer.data

    @classmethod
    def create(cls, xml_path: str, work_piece: Material):
        self = cls()
        self.work_piece = work_piece
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found at {xml_path}")

        spec = mujoco.MjSpec.from_file(xml_path)

        # 1. Register Mesh 1
        mesh1 = spec.add_mesh(name="mesh1", file=SCALPEL_HANDLER_1_PATH)
        mesh1.refquat = [0.707, -0.707, 0, 0]
        mesh1.scale = [0.001, 0.001, 0.001]

        # 2. Register Mesh 2
        mesh2 = spec.add_mesh(name="mesh2", file=SCALPEL_HANDLER_2_PATH)
        mesh2.refquat = [0.0, 1.0, 0, 0] # Match the orientation of part 1
        mesh2.scale = [0.001, 0.001, 0.001]
        # 3. Register the scalpel mesh
        mesh3 = spec.add_mesh(name="mesh3", file=SCALPEL_PATH)
        #  Rotate 60 degrees To mach scalpel being perpendicular to ee
        mesh3.refquat = [0, 0.5, 0, 0.866]
        mesh3.scale = [0.0001, 0.0001, 0.0001]
        # 4. Setup Body Hierarchy
        ee_body = spec.body("link7")
        attach_site = spec.site("attachment_site")

        handler = ee_body.add_body(name="3d_printed_handler")
        # Align handler origin to the attachment site
        handler.pos = attach_site.pos
        handler.quat = attach_site.quat
        handler_depth = -0.09
        # 5. Add Part 1 (First half of handler)
        handler.add_geom(
            name="handler_part1_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="mesh1",
            rgba=[0.9, 0.9, 0.0, 1],  # Orange
            pos=attach_site.pos + [-0.045, 0.045, handler_depth]
        )

        # 6. Part 2 (Second half of handler)
        # We add this to the SAME body so they are fixed together.
        handler.add_geom(
            name="handler_part2_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="mesh2",
            rgba=[0.8, 0.8, 0.8, 1],  # Grey to distinguish
            pos=attach_site.pos + [-0.0175, -0.002, 0.025]
        )
        # 7. Scalpel geometry from registered scalpel mesh
        handler.add_geom(
            name=SCALPEL_GEOM,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="mesh3",
            rgba=[0.8, 0.8, 0.8, 1],  # Grey to distinguish
            pos=attach_site.pos,
        contype=1,       # Bitmask: belongs to group 1
        conaffinity=2 if self.work_piece.is_solid else 1    # Bitmask: only collides with group 1
        )


        # 8. TCP defined at the tip of scalpel
        tip_offset = [0, 0, 0.113]
        tcp_site = handler.add_site(
            name="scalpel_tip",
            pos=attach_site.pos + tip_offset,
            size=[0.002, 0.002, 0.002],  # Small visual marker
            rgba=[1, 0, 0, 1],  # Red tip
            group=1  # Ensure group 1 is enabled in viewer
        )
        # 9. Update Material to use Collision Group 2
        material = spec.worldbody.add_body(name=self.work_piece.name)
        # Added a slide joint so the material can move up/down
        material.add_joint(
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],          # vertical axis
            name="material_slide",
            limited=True,
            range=[-0.1, 0.1]        # adjust as needed
        )
        material.add_geom(
            name=MATERIAL_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.work_piece.size,
            pos=self.work_piece.center,
            rgba=[0.2, 0.8, 0.2, 0.7],
            contype=2,       # Bitmask: belongs to group 2
            conaffinity=1 if self.work_piece.is_solid else 2,   # Bitmask: only collides with group 2
            solref=[0.02, 1],
            margin=0.001
        )
        # 10. Compile model, forward it with data
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self.buffer = SharedMemoryBuffer(
            name=None,                       # anonymous
            create=True,
            num_signals=6,
            buffer_size=1000,                 # use your config value
            signal_names=["Fx (N)", "Fy (N)", "Fz (N)", "Kdx", "Kdy", "Kdz"]
        )
        # self.robot_state["shared_memory"] = self.buffer.data
        self.set_shm_buffer()

        Logger.info("✓ Model created")
        Logger.info(f"Scalpel geom ID: {self._model.geom(SCALPEL_GEOM).id}")
        Logger.info(f"Material geom ID: {self._model.geom(MATERIAL_GEOM).id}")
        return self

    def shutdown(self):
        if hasattr(self, 'robot_state') and 'shared_array' in self.robot_state:
            self.robot_state['shared_array'] = None   # release reference
        if self.buffer:
            self.buffer.close()
            self.buffer.unlink()

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def shm(self):
        # For backward compatibility with existing code that expects self.shm
        # (like the main script passing robot.shm.name)
        return self.buffer

    def reset_state(self):
        """Reset the state dictionary"""
        if not self.robot_state.get(FORCE_HISTORY):
            self.robot_state[FORCE_HISTORY] = []  # Store cutting forces
        else:
            self.robot_state.get(FORCE_HISTORY).clear()

    def experiment_exeucte_wrapper(self, callback: Callable[[], None], *args):
        if callable(callback):
            return callback(*args)
        else:
            print("callback must be a Callable")
            return 1


# Run the experiment
if __name__ == "__main__":
    pass
