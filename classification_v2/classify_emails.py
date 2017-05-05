#!_*_ coding: utf8 _*_
text1 = "Se eu comprar cinco anos antecipados, eu ganho algum desconto?"
text2 = "O exercicio 15 do curso de Java 1 está com a resposta errada. Pode conferir pf?"
text3 = "Existe algum curso para cuidar do marketing da minha empresa?"

import pandas as pd

classification = pd.read_csv('emails.csv')

textPure = classification['email']
textBroke = textPure.str.split(' ')

print textBroke