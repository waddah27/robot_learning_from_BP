from startup import *
from visualizer import Visualize
from base import Robot
# 1 - MjModel (mjModel)
if CONF["use_iiwa14"]:
  kuka_iiwa_14 = os.path.join(os.path.dirname(__file__), 'kuka_iiwa_14')
  iiwa14_xml = os.path.join(kuka_iiwa_14, 'scene.xml')
  
  robot = Robot()
  robot.create(iiwa14_xml)
  
  print(robot.model.opt.gravity)
 
# 2 - MjData (mjData)

visualizer = Visualize(robot)

if __name__=='__main__':
  
    
    # Rendering 
    # model.geom('red_box').rgba[:3] = np.random.rand(3)
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
  


