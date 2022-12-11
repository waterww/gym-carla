import os
import time

from carla_server import CarlaServer
from carla_train import main_test 

workpath = os.getcwd()
HPC = True
if "miao" in workpath:
    CarlaServer.CARLA_ROOT = "/home/miao/CARLA_0.9.6"
    HPC = False # if not HPC, offscreen = False
else:
    CarlaServer.CARLA_ROOT = os.path.join(os.environ['VSC_DATA'], 'lib', 'carla')

CarlaServer.CARLA_VERSION = (0, 9, 6)


def test_carla(client):
    # Test CARLA by retrieving image from camera sensor
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    # Enable synchronous mode with fixed timestep
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    # Setup camera sensor
    spawn_point = world.get_map().get_spawn_points()[0]
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, spawn_point)
    camera.listen(lambda img: img.save_to_disk('camera_image.png'))

    # World tick, wait for camera image and cleanup
    world.tick()
    time.sleep(5)
    camera.destroy()

    print("test carla finished!")


if __name__ == "__main__":
    # Launch server and connect client
    myport = 2222
    myepisodes = 5000
    print("carla port: %d" % myport)
    separate_client_process = False

    with CarlaServer(myport, offscreen=HPC) as server:
        if separate_client_process:
            # New client connection and test_carla method are executed in a separate process
            server.run_client(main_test(port=myport, episodes=myepisodes))
        else:
            # New client connection and test_carla method are executed in current process
            client = server.connect_client()
            main_test(port=myport, episodes=myepisodes)
