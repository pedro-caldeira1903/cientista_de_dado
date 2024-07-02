!pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
consumo_açucar=ctrl.Antecedent(np.arange(0, 20, 0.1), 'consumo_açucar')
pratica_exercicio=ctrl.Antecedent(np.arange(0, 20, 0.1), 'pratica_exercicio')
diabetes=ctrl.Consequent(np.arange(0, 100, 0.1), 'diabetes')
consumo_açucar.automf(3, variable_type='quant')
pratica_exercicio.automf(3, variable_type='quant')
diabetes['low']=fuzz.sigmf(diabetes.universe, 20, -0.2)
diabetes['average']=fuzz.gaussmf(diabetes.universe, 40, -10)
diabetes['high']=fuzz.pimf(diabetes.universe, 40, 100, 180, float('inf'))
regra1=ctrl.Rule(consumo_açucar['high'] | pratica_exercicio['low'], diabetes['high'])
regra2=ctrl.Rule(consumo_açucar['average'], diabetes['average'])
regra3=ctrl.Rule(consumo_açucar['low'] | pratica_exercicio['high'], diabetes['low'])
sistema=ctrl.ControlSystem([regra1, regra2, regra3])
resultado=ctrl.ControlSystemSimulation(sistema)
resultado.input['consumo_açucar']=int(input('Digite o consumo de açucar: '))
resultado.input['pratica_exercicio']=int(input('Digite a sua pratica exercicio: '))
resultado.compute()
print('A probabilidade de ter diabetes é', resultado.output['diabetes'],'%')
