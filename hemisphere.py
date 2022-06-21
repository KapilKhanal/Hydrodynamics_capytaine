

#"Analytical(A.Hulme) and Capytaine Comparison
import os
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import pygmsh
import math
import logging

import matplotlib.pyplot as plt
import capytaine as cpt
from capytaine import FloatingBody
from capytaine.post_pro import rao
import meshmagick
from scipy.linalg import block_diag
import meshmagick.mesh as mm

from packaging import version
if version.parse(meshmagick.__version__) < version.parse('3.0'):
	import meshmagick.hydrostatics as hs
else:
	import meshmagick.hydrostatics_old as hs

from capytaine.io.xarray import separate_complex_values
from scipy.optimize import minimize
from scipy import integrate


  



def compare_mass_dampings():
	rho_sw = 1023 #kg/m^3
	alpha =0.5 
	rho_structure = 650
	g = 9.81

	K = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,7,8,9,10])
	body = cpt.Sphere(radius = 1,clip_free_surface=True, ntheta=40, nphi=40)
	body.add_all_rigid_body_dofs()
	body.keep_immersed_part(inplace = True)

	#body.show_matplotlib()

	wave_direction=0.0
	#use meshmagick to compute hydrostatic stiffness 
	hsd = hs.Hydrostatics(mm.Mesh(body.mesh.vertices, body.mesh.faces),
	cog=(0,0,0),
	rho_water=rho_sw,
	grav=g).hs_data

	m = hsd['disp_mass']

	I = np.array([[hsd['Ixx'], -1*hsd['Ixy'], -1*hsd['Ixz']],[-1*hsd['Ixy'], hsd['Iyy'], -1*hsd['Iyz']],[-1*hsd['Ixz'], -1*hsd['Iyz'], hsd['Izz']]])
	M = block_diag(m,m,m,I)
	body.mass = body.add_dofs_labels_to_matrix(M)
	#print(body.mass)


	# Hydrostatics
	kHS = block_diag(0,0,hsd['stiffness_matrix'],0)
	body.hydrostatic_stiffness = body.add_dofs_labels_to_matrix(kHS)
	problems = [
	    cpt.RadiationProblem(body = body, radiating_dof="Heave", omega=omega, rho=rho_sw)
	    for omega in K
	]
	# problems += [
	#     cpt.DiffractionProblem(body = body, omega=omega, rho=rho_sw)
	#     for dof in body.dofs
	#     for omega in omega_range
	# ]
	print(f"RADIUS MAX : {8*problems[0].body.mesh.faces_radiuses.max()}")


	# Solve all problems
	direct_linear_solver = cpt.BasicMatrixEngine(linear_solver='direct')
	solver = cpt.BEMSolver(engine=direct_linear_solver)
	results = [solver.solve(pb, keep_details = True) for pb in sorted(problems)]


	# Gather the computed added mass into a labelled array.
	dataset = cpt.assemble_dataset(results, wavenumber = True)

	#plotting addded mass and damping for each wave number
	added_mass = [r.added_masses['Heave'] for r in results]
	# normalizing with 2*pi*rho*R^3/3
	added_mass = [(3*r)/(2*np.pi*rho_sw*1) for r in added_mass] 
	


	
	analytical_mass = [0.8764,0.8627,0.7938,0.7157,0.6452,0.5861,0.5381,0.4999,0.4698,
0.4464,0.4284,0.4047,0.3924,0.3871,0.3864,0.3884,0.3988,0.4111,0.4322,0.4471,0.4574,0.4647,0.47,0.474,0.4771]
	rad_dampings = np.array([r.radiation_dampings['Heave'] for r in results])
	#normalizing with 2*pi*rho*R^3*omega.
	rad_dampings  = (3*rad_dampings)/(2*np.pi*rho_sw*1*K)
	analy_damping = [0.1036,0.1816,0.2793,0.3254,0.341,0.3391,0.3271,0.3098,0.2899,0.2691,0.2484,0.2096,0.1756,0.1469,0.1229,0.1031,
0.0674,0.0452,0.0219,0.0116,0.0066,0.004,0.0026,0.0017,0.0012]
	wave_number = dataset.wavenumber
	
	plt.plot(K, added_mass, "red", label = "Added Mass(A)")
	plt.plot(K, analytical_mass, "red", label = "Analytical Mass(A)", marker = 's',linestyle='dashed')
	plt.plot(K, rad_dampings, "blue", label = "Damping(B)", fillstyle = 'none')
	plt.plot(K, analy_damping, "blue", label = "Analytical Damping(B)", marker = 'o',linestyle='dashed')
	plt.xlabel("Wave number (K)")
	plt.ylabel("A, B")
	plt.legend(loc='best')
	
	plt.show()
	return results



if __name__ == '__main__':
	compare_mass_dampings()

