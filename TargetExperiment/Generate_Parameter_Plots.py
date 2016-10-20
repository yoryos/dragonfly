import numpy as np
import ast
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.interpolate import spline


def read_test_log(file):
    tests = []
    with open(file) as f:
        for line in f:
            index_dict = line.split("{")
            index = index_dict[0]
            d = "{" + index_dict[1]
            d = ast.literal_eval(d)
            tests.append((index.strip(), d))
    return tests


def test_has_settings(test, settings):
    for setting, value in settings:
        try:
            if test[setting] != value:
                return False
        except:
            return False

    return True


def get_tests_with_constant_settings(tests, settings):
    if settings is []:
        return tests
    good_tests = []
    for index, test in tests:
        if test_has_settings(test, settings):
            good_tests.append(index)
    return good_tests


def get_test_data(index, top_path, folder_prefix, file_prefix, num=3):
    data = []
    for i in xrange(num):
        data.append(np.loadtxt(os.path.join(top_path, folder_prefix + str(index), file_prefix + str(i + 1) + ".dat")))

    return data


def cost_function(first, second, both, c = 0):

    return min(abs(both - first), abs(both - second)) - c * max(abs(both - first), abs(both - second))


def cost_test(data, weight):
    both, lr, tb = data
    total_cost = 0
    somas_cost = []
    soma_total_cost = []
    for j in xrange(lr.shape[1]):
        soma_cost = np.zeros(lr.shape[0])
        for i in xrange(lr.shape[0]):
            soma_cost[i] = cost_function(lr[i, j], tb[i, j], both[i, j], weight)
        s = soma_cost.sum()
        soma_total_cost.append(s)
        total_cost += s
        somas_cost.append(soma_cost)

    return total_cost, soma_total_cost, somas_cost


def get_test_from_index(tests, index):
    for i, test in tests:
        if i == index:
            return test
    return None


def plot_data(index, path):
    plt.figure()
    d = get_test_data(index, path, "targetExperiment_", "rates_")
    grand_total, total, somas = cost_test(d,0)
    grand_total2, total2, somas2 = cost_test(d,1)
    for i in xrange(1, len(somas)):
        plt.subplot(len(somas) - 1, 1, i)
        if i == 1:
            plt.title("Total error over all somas for " + str(index) + " = " + str(grand_total) + " and alternative " +
                      str(grand_total2))
        plt.ylabel("Soma error" + str(total[i]))
        plt.plot(d[0][:, i])
        plt.plot(d[1][:, i])
        plt.plot(d[2][:, i])
        plt.plot(somas[i])
        plt.plot(somas2[i])
    plt.legend(["Both", "LeftRight", "TopBottom", "Cost1", "Costs2"])


def get_costs_for_tests(tests, path, weight=0):
    costs = []
    for test in tests:
        d = get_test_data(test, path, "targetExperiment_", "rates_")
        c, _, _ = cost_test(d, weight)
        costs.append(c)

    return costs


def get_mesh(settings, test_with_settings, tests, costs):
    s = []
    for test in test_with_settings:
        s.append((test, get_test_from_index(tests, test)))

    variable_settings = []
    for c, (i, j) in enumerate(s):
        test_settings = []
        for key in j.keys():
            if key not in [k[0] for k in settings]:
                if isinstance(j[key], str):
                    j[key] = int(j[key].split(".")[0])
                test_settings.append((key, j[key]))
        variable_settings.append((i, test_settings, costs[c]))

    print
    #
    # for i, j, c in variable_settings:
    #     print i, j, c


    X = []
    Y = []
    Z = []
    I = []
    for i, settings, cost in variable_settings:
        X.append(settings[0][1])
        Y.append(settings[1][1])
        Z.append(cost)
        I.append(str(i))

    x_l = variable_settings[0][1][0][0]
    y_l = variable_settings[0][1][1][0]
    l = np.where(np.array(X)[:-1] != np.array(X)[1:])[0][0] + 1
    S = [len(X) / l, l]
    X_mesh = np.array(X).reshape(S)
    Y_mesh = np.array(Y).reshape(S)
    Z_mesh = np.array(Z).reshape(S)
    return X_mesh, Y_mesh, Z_mesh, X, Y, Z, I, x_l, y_l


settings = [('synapse_max_conductance', 1.0),
            ('estmd_gain', 10.0)]
#
# settings = [('synapses_file_name', '1000.dat'),
#             ('estmd_gain', 10.0)]
#
# settings = [('tau_gaba', 400),
#             ('estmd_gain', 10.0)]

variable_parameters = []
# path = 'Parameter_Data/pSearch_3_lm1015_2016-04-21_20:50:49'

settings = []
path = 'Parameter_Data/pSearch_2_cps15_2016-05-10_12:10:45'
# path = "Parameter_Data/pSearch_3_cps15_2016-05-10_11:53:50"
# path = "Parameter_Data/pSearch_3_cps15_2016-05-11_01:45:27"
# path = "Parameter_Data/Combined"
tests = read_test_log(os.path.join(path, 'parameter_index.txt'))
test_with_settings = get_tests_with_constant_settings(tests, settings)

from matplotlib import cm
fig = plt.figure()

def plot_mesh(weight, tests, test_with_settings, i, mesh = False, l_X = False):

    costs = get_costs_for_tests(test_with_settings, path, weight=weight)
    X_mesh, Y_mesh, Z_mesh, X, Y, Z, I, x_l, y_l = get_mesh(settings, test_with_settings, tests, costs)
    Z_mesh /= 10e3
    Z = [j/10e3 for j in Z]
    ax = plt.subplot(i)#, projection='3d')
    # surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh,cmap=cm.coolwarm, rstride=1, cstride=1,
    #                    linewidth=0, antialiased=False, alpha = 0.6)
    print X_mesh
    print Y_mesh
    if x_l == "synapses_file_name":
        x_l = "no. synapses"
    if y_l == "synapses_file_name":
        y_l = "no. synapses"
    if x_l == "tau_gaba":
        x_l = r"$\tau_{gaba}$"
    if y_l == "tau_gaba":
        y_l = r"$\tau_{gaba}$"
    if x_l == "synapse_max_conductance":
        x_l = r"$\bar{g}_{syn}$"
    if y_l == "synapse_max_conductance":
        y_l = r"$\bar{g}_{syn}$"
    if x_l == "e_gaba":
        x_l = r"$E_{gaba}$"
    if y_l == "e_gaba":
        y_l = r"$E_{gaba}$"

    if l_X and not mesh:
        for i in xrange(X_mesh.shape[0]):
            ax.plot(Y_mesh[i,:],Z_mesh[i,:],label = x_l + " = " + str(X_mesh[i][0]))
        ax.set_xlabel(y_l)
        ax.set_ylabel(r"$J_{bistable}($" + "%0.2f"%(weight) + r"$) / 10e3$")
    elif not mesh:
        for i in xrange(Y_mesh.shape[1]):
        # xnew = np.linspace(X_mesh[:,i][0],X_mesh[:,i][-1],10)
        # power_smooth = spline(X_mesh[:,i],Z_mesh[:,i],xnew)
        # plt.plot(xnew,power_smooth)
            ax.plot(X_mesh[:,i],Z_mesh[:,i],label = y_l + " = " + str(Y_mesh[0][i]))
        ax.set_xlabel(x_l, fontsize = 18)
        ax.set_ylabel(r"$J_{bistable}($" + "%0.2f"%(weight) + r"$) / 10e3$", fontsize = 18)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
    else:
    #
        surf = ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh)
        ax.contour(X_mesh, Y_mesh, Z_mesh, zdir='z', offset=min(Z) - 50, cmap=cm.coolwarm)
        ax.scatter(X, Y, Z)
        for i in xrange(len(X)):
            ax.text(X[i], Y[i], Z[i], I[i])
        ax.set_xlabel(x_l)
        ax.set_ylabel(y_l)
        ax.set_zlabel(r"$J_{bistable}($" + "%0.2f"%(weight) + r"$) / 10e3$")
    # for i in xrange(len(X)):
    #     ax.text(X[i], Z[i], I[i])


    highest_x = "%0.2f"%(X[Z.index(max(Z))])
    lowest_x =  "%0.2f"%(X[Z.index(min(Z))])
    highest_y = "%0.2f"%(Y[Z.index(max(Z))])
    lowest_y =  "%0.2f"%(Y[Z.index(min(Z))])
    highest = I[Z.index(max(Z))]
    lowest = I[Z.index(min(Z))]
    print weight,highest, lowest

    # plt.title("Worst (" + str(x_l) + ":" + str(highest_x) + " " + str(y_l) + ":" + str(highest_y) + " trial " + str(
    #         highest) +
    #           ") \n Best (" + str(x_l) +":" + str(lowest_x) + " " + str(y_l) + ":" + str(lowest_y) + " trial " + str(
    #         lowest) + ")", fontsize =  15)

    return ax



import matplotlib.gridspec as gridspec
T = 3
t = int(np.ceil(np.sqrt(T)))
print t
gs1 = gridspec.GridSpec(2,3)
gs1.update(wspace=0.155, hspace=0.17)
w = np.linspace(0,0.6,T)
print w
for i in xrange(T):

    ax = plot_mesh(w[i], tests, test_with_settings, gs1[i], mesh=True)

h,l = ax.get_legend_handles_labels()
fig.legend(h,l,loc='lower center', ncol=5, borderaxespad=25, fontsize = 16)
plt.show()

# plot_data(154,path)
# plot_data(31,path)

# print costs
# print len(test_with_settings)

# for i,test in tests:
#     print i,test
