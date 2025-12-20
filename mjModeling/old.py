from startup import mujoco, CONF
xml = """
  <mujoco>
    <worldbody>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </worldbody>
  </mujoco>
  """
xml = """
  <mujoco>
    <worldbody>
      <light name="top" pos="0 0 1"/>
      <body name="box_and_sphere" euler="0 0 -30">
        <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
      </body>
    </worldbody>
  </mujoco>
  """
model = mujoco.MjModel.from_xml_string(xml)
print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, 10)
print('flipped gravity', model.opt.gravity)
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)
if not CONF["use_iiwa14"]:
    print("Some model specifications:")
      
    print("Num of geometries : %s"%model.ngeom)
    print(model.geom_rgba)
    try:
        print("All geometries: %s"%model.geom())
    except KeyError as e:
        print(e)
          
    print(model.geom('green_sphere').rgba)
    print('id of "green_sphere": ', model.geom('green_sphere').id)
    print('name of geom 1: ', model.geom(1).name)
    print('name of body 0: ', model.body(0).name)
      
    geom_names = [model.geom(i).name for i in range(model.ngeom)]  
    print("geometries' names : %s"%geom_names)      

    print("Some data specifications:")
    print(data.geom_xpos)
      
    print('raw access:\n', data.geom_xpos)

    # MjData also supports named access:
    print('\nnamed access:\n', data.geom('green_sphere').xpos)