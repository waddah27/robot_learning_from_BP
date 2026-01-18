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
