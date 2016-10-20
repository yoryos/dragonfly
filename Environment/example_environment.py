from Helper.Configurer import EnvironmentConfigurer
from Environment.Environment import Environment
import datetime
import time

environment_configerer = EnvironmentConfigurer("Environment/example_environment.ini")
env_general = environment_configerer.config_section_map("General")
_, t = environment_configerer.get_targets()

start_time = time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')

environment = Environment(background_path=env_general["background"],
                          dt=env_general["dt"],
                          ppm=env_general["ppm"],
                          width=env_general["width"],
                          height=env_general["height"],
                          run_id=st,
                          target_config=t)

for i in xrange(750):
    _,r = environment.step((5.0,0,0,0.0))
    print environment