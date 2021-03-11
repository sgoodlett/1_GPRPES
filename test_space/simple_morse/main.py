import peslearn
import numpy as np

input_string = ("""
H
H 1 r1 

r1 = [0, 5, 1000000]

remove_redundancy = true
input_name = input.dat        
""")

input_object = peslearn.InputProcessor(input_string)
template_object = peslearn.datagen.Template("./template.dat")
molecule_object = peslearn.datagen.Molecule(input_object.zmat_string)
#config = peslearn.datagen.ConfigurationSpace(molecule_object, input_object)
#config.generate_PES(template_object)

input_object.set_keyword({'energy':'regex'})
input_object.set_keyword({'energy_regex':'   Total Energy\s+=\s+(-\d+\.\d+)'})
input_object.set_keyword({'pes_format':'interatomics'})

#peslearn.utils.parsing_helper.parse(input_object, molecule_object)

input_object.set_keyword({'use_pips':'false'})
input_object.set_keyword({'training_points':800000})
input_object.set_keyword({'sampling':'random'})
input_object.set_keyword({'hp_maxit':10})
input_object.set_keyword({'rseed':0})

svigp = peslearn.ml.SVIGP("sine.dat", input_object, molecule_type='A2', batchsize = 2000, inducing_n = 100, max_iter = 100)
svigp.optimize_model()

#gp = peslearn.ml.GaussianProcess("sine.dat", input_object, molecule_type='A2')
#gp.optimize_model()


