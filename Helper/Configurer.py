import ConfigParser

import numpy as np
import os

class Configurer(object):
    """Class for loading configuration files to create BrainModules"""

    def __init__(self, path, defaults=None):
        self.config = ConfigParser.SafeConfigParser(allow_no_value=True, defaults=defaults)
        if not os.path.isfile(path):
            print "Could not find config file: " + path
            raise IOError
        self.config.read(path)
        print "Found the following sections in the config file: " + path + "\n \t->", self.config.sections()

    def config_section_map(self, section, vars=None):
        dict1 = {}
        options = self.config.options(section)
        for option in options:
            bool_index = option.find("(bool)")
            float_index = option.find("(float)")
            int_index = option.find("(int)")
            list_index = option.find("(list)")
            option_name = option
            try:
                if bool_index > 0:
                    option_name = option[:bool_index]
                    dict1[option_name] = self.config.getboolean(section, option)
                elif float_index > 0:
                    option_name = option[:float_index]
                    dict1[option_name] = self.config.getfloat(section, option)
                elif int_index > 0:
                    option_name = option[:int_index]
                    dict1[option_name] = self.config.getint(section, option)
                elif list_index > 0:
                    option_name = option[:list_index]
                    list_string = self.config.get(section,option)
                    l = list_string.split(",")
                    ind = l[0].rfind("[")
                    l[0] = l[0][ind+1:]
                    l[-1] = l[-1][:-ind-1]
                    final_list = [float(i) for i in l]
                    for i in xrange(ind):
                        final_list = [final_list]

                    dict1[option_name] = final_list
                else:
                    dict1[option] = self.config.get(section, option, vars=vars)

                if dict1[option_name] == -1:
                    print("skip: %s " % option)

            except Exception as inst:
                print "Could not get", option, "set to None/False",
                if bool_index > 0:
                    option_name = option[:bool_index]
                    dict1[option_name] = False
                else:
                    dict1[option_name] = None

            if vars is not None:
                assert isinstance(vars, dict)
                if option_name in vars.keys():
                    dict1[option_name] = vars[option_name]

        return dict1


class EnvironmentConfigurer(Configurer):
    def __init__(self, path, defaults=None):
        Configurer.__init__(self, path, defaults)

    def get_targets(self, section_prefix="Target_"):
        targets = []

        all_sections = self.config.sections()
        target_sections = [section for section in all_sections if
                           str(section).startswith(section_prefix, 0, len(section_prefix))]

        print "Found target sections: ", target_sections

        for target_section in target_sections:
            target = {}

            try:
                velocity = self.config.get(target_section, "velocity")
                if velocity is not None:
                    velocity_list = velocity.split(",")
                    target['velocity'] = np.asarray([float(v) for v in velocity_list])

                wobble = self.config.get(target_section, "wobble")
                if wobble is not None:
                    target['wobble'] = float(wobble)

                position = self.config.get(target_section, "position")
                if position is not None:
                    position_list = position.split(",")
                    target['position'] = np.asarray([float(p) for p in position_list])

                size = self.config.get(target_section, "size")
                if size is not None:
                    target['size'] = float(size)

                color = self.config.get(target_section, "color")
                if color is not None:
                    color_list = color.split(",")
                    target['color'] = [float(h) for h in color_list]

                print target_section, "config: ", target

            except Exception, e:
                print "Incorrect target data format for " + target_section
                print str(e)
                return False, []

            targets.append(target)

        return True, targets
