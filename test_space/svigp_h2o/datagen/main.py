import peslearn
import numpy as np

input_string = ("""
O 
H 1 r1
H 1 r2 2 a2 

r1 = [0.85,1.20, 5]
r2 = [0.85,1.20, 5]
a2 = [90.0,120.0, 5]

remove_redundancy = true
input_name = input.dat        
""")

input_object = peslearn.InputProcessor(input_string)
template_object = peslearn.datagen.Template("./template.dat")
molecule_object = peslearn.datagen.Molecule(input_object.zmat_string)
config = peslearn.datagen.ConfigurationSpace(molecule_object, input_object)
config.generate_PES(template_object)

input_object.set_keyword({'energy':'regex'})
input_object.set_keyword({'energy_regex':'   Total Energy\s+=\s+(-\d+\.\d+)'})
input_object.set_keyword({'pes_format':'interatomics'})

#peslearn.utils.parsing_helper.parse(input_object, molecule_object)

input_object.set_keyword({'use_pips':'true'})
input_object.set_keyword({'training_points':40})
input_object.set_keyword({'sampling':'structure_based'})
input_object.set_keyword({'hp_maxit':10})
input_object.set_keyword({'rseed':0})

