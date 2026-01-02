from mjModeling.iiwa14_model import iiwa14
from mjModeling import *
from visualizer import Visualize

# 1 - build experiment env
robot = iiwa14()
robot.create(robot_scene_xml)
  
print(robot.model.opt.gravity)
 
# 2 - simulator
visualizer = Visualize(robot)

if __name__=='__main__':
  
    
    # Rendering 
  if CONF["run_one_frame"]:
      with mujoco.Renderer(robot.model) as renderer:
          # first we do forward kins to compute the accelerations (derivatives of the state)
          mujoco.mj_forward(robot.model, robot.data)
          renderer.update_scene(robot.data)
          image_array = renderer.render()
          # img = Image.fromarray(image_array)
          # img.show()
          plt.imshow(image_array)
          plt.axis("off")
          plt.title("Robot")
          plt.show()
  else:
    visualizer.run()
  


