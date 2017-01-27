
# coding: utf-8

# # Nanodegree Engenheiro de Machine Learning
# ## Aprendizagem Supervisionada
# ## Project 2: Construindo um Sistema de Intervenção para Estudantes

# Bem-vindo ao segundo projeto do Nanodegree de Machine Learning! Neste Notebook, alguns templates de código já foram fornecidos, e será o seu trabalho implementar funcionalidades necessárias para completar este projeto com êxito. Seções que começam com **'Implementação'** no cabeçalho indicam que o bloco de código que se segue precisará de funcionalidades adicionais que você deve fornecer. Instruções serão providenciadas para cada seção e as especificações para cada implementação estarão marcadas no bloco de código com o comando `'TODO'`. Tenha certeza de ler atentamente todas as instruções!
# 
# Além do código implementado, haverá questões relacionadas ao projeto e à implementação que você deve responder. Cada seção em que você tem que responder uma questão será antecedida de um cabeçalho **'Questão X'**. Leia atentamente cada questão e escreva respostas completas nas caixas de texto subsequentes que começam com **'Resposta: '**. O projeto enviado será avaliado baseado nas respostas para cada questão e a implementação que você forneceu.  
# 
# >**Nota:** Células de código e Markdown podem ser executadas utilizando o atalho de teclado **Shift + Enter**. Além disso, as células Markdown podem ser editadas, um clique duplo na célula entra no modo de edição.

# ### Questão 1 - Classificação versus Regressão
# *Seu objetivo neste projeto é identificar estudantes que possam precisar de intervenção antecipada antes de serem reprovados. Que tipo de problema de aprendizagem supervisionada é esse: classificação ou regressão? Por quê?*

# **Resposta: ** Esse algoritimo de aprendizagem pertence à classificação, seguindo o argumento com base na descrição de classificação, ou seja atravez de uma relação conhefico identificar a quais classes os dados pertence. De forma que o algoritimo de aprendizagem saiba identificar se o aluno vai ser formar ou não.

# ## Observando os Dados
# Execute a célula de código abaixo para carregar as bibliotecas de Python necessárias e os dados sobre os estudantes. Note que a última coluna desse conjunto de dados, `'passed'`, será nosso rótulo alvo (se o aluno foi ou não aprovado). As outras colunas são atributos sobre cada aluno.

# In[13]:

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Os dados dos estudantes foram lidos com êxito!"


# ### Implementação: Observando os Dados
# Vamos começar observando o conjunto de dados para determinar quantos são os estudantes sobre os quais temos informações e entender a taxa de graduação entre esses estudantes. Na célula de código abaixo, você vai precisar calcular o seguinte:
# - O número total de estudantes, `n_students`.
# - O número total de atributos para cada estudante, `n_features`.
# - O número de estudantes aprovados, `n_passed`.
# - O número de estudantes reprovados, `n_failed`.
# - A taxa de graduação da classe, `grad_rate`, em porcentagem (%).
# 

# In[14]:

n_students = len(student_data.index)
n_features = len(student_data.columns)-1
n_passed = student_data.passed.str.contains('yes').sum()
n_failed = student_data.passed.str.contains('no').sum()
grad_rate = (float(n_passed)/float(n_students))*100

print "Número total de estudantes: {}".format(n_students)
print "Número de atributos: {}".format(n_features)
print "Número de estudantes aprovados: {}".format(n_passed)
print "Número de estudantes reprovados: {}".format(n_failed)
print "Taxa de graduação: {:.2f}%".format(grad_rate)


# ## Preparando os Dados
# Nesta seção, vamos preparara os dados para modelagem, treinamento e teste.
# 
# ### Identificar atributos e variáveis-alvo
# É comum que os dados que você obteve contenham atributos não numéricos. Isso pode ser um problema, dado que a maioria dos algoritmos de machine learning esperam dados númericos para operar cálculos.
# 
# Execute a célula de código abaixo para separar os dados dos estudantes em atributos e variáveis-alvo e verificar se algum desses atributos é não numérico.

# In[5]:

feature_cols = list(student_data.columns[:-1])

target_col = student_data.columns[-1] 

print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

X_all = student_data[feature_cols]
y_all = student_data[target_col]

print "\nFeature values:"
print X_all.head()


# ### Pré-processar Colunas de Atributo
# 
# Como você pode ver, há muitas colunas não numéricas que precisam ser convertidas! Muitas delas são simplesmente `yes`/`no`, por exemplo, a coluna `internet`. É razoável converter essas variáveis em valores (binários) `1`/`0`.
# 
# Outras colunas, como `Mjob` e `Fjob`, têm mais do que dois valores e são conhecidas como variáveis categóricas. A maneira recomendada de lidar com esse tipo de coluna é criar uma quantidade de colunas proporcional aos possíveis valores (por exemplo, `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc), e assinalar `1` para um deles e `0` para todos os outros.
# 
# Essas colunas geradas são por vezes chamadas de _variáveis postiças_ (em inglês: _dummy variables_), e nós iremos utilizar a função [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) para fazer essa conversão. Execute a célula de código abaixo para executar a rotina de pré-processamento discutida nesta seção.

# In[15]:


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():
        
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Implementação: Divisão dos Dados de Treinamento e Teste
# Até agora, nós convertemos todos os atributos _categóricos_ em valores numéricos. Para o próximo passo, vamos dividir os dados (tanto atributos como os rótulos correspondentes) em conjuntos de treinamento e teste. Na célula de código abaixo, você irá precisar implementar o seguinte:
# - Embaralhe aleatoriamente os dados (`X_all`, `y_all`) em subconjuntos de treinamento e teste.
#   - Utilizar 300 pontos de treinamento (aproxidamente 75%) e 95 pontos de teste (aproximadamente 25%).
#   - Estabelecer um `random_state` para as funções que você utiliza, se a opção existir.
#   - Armazene os resultados em `X_train`, `X_test`, `y_train` e `y_test`.

# In[16]:

from sklearn.cross_validation import train_test_split

num_train = 300

num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=num_train, test_size=num_test, random_state=42)

print "O conjunto de treinamento tem {} amostras.".format(X_train.shape[0])
print "O conjunto de teste tem {} amostras.".format(X_test.shape[0])


# ## Treinando e Avaliando Modelos
# Nesta seção, você irá escolher 3 modelos de aprendizagem supervisionada que sejam apropriados para esse problema e que estejam disponíveis no `scikit-learn`. Primeiro você irá discutir o raciocínio por trás da escolha desses três modelos considerando suas vantagens e desvantagens e o que você sabe sobre os dados. Depois você irá ajustar o modelo a diferentes tamanhos de conjuntos de treinamento (com 100, 200 e 300 pontos) e medir a pontuação F<sub>1</sub>. Você vai precisar preencher três tabelas (uma para cada modelo) que mostrem o tamanho do conjunto de treinamento, o tempo de treinamento, o tempo de previsão e a pontuação F<sub>1</sub> no conjunto de treinamento.
# 
# **Os seguintes modelos de aprendizagem supervisionada estão atualmente disponíveis no **[`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html)** para você escolher:**
# - Gaussian Naive Bayes (GaussianNB)
# - Árvores de Decisão
# - Métodos de agregação (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Método do gradiente estocástico (SGDC)
# - Máquinas de vetores de suporte (SVM)
# - Regressão logística

# ### Questão 2 - Aplicação dos Modelos
# *Liste três modelos de aprendizagem supervisionada que são apropriadas para esse problema. Para cada modelo escolhido:*
# - Descreva uma aplicação em mundo real na indústria em que o modelo pode ser aplicado. *(Talvez você precise fazer um pouco de pesquisa para responder essa questão – dê as devidas referências!)* 
# - Quais são as vantagens do modelo; quando ele tem desempenho melhor? 
# - Quais são as desvantagens do modelo, quando ele tem desempenho pior?
# - O que faz desse modelo um bom candidato para o problema, considerando o que você sabe sobre os dados?

# **Resposta: ** Durante a pesquisa percebi que para esse exemplo os modelos mais assertivos são: Maquinas de Vetores de Suporte (SVM), Árvores de Decisão (DT) e Naive Bayes (NB). A escolha desses modelos foi baseada na eficiencia desses algoritimos de aprendizagem para trabalhar com classificação e também para pequenos conjuntos de dados com bom desenpenho.
# 
# ####  Maquinas de Vetores de Suport  (SVM)
# 
# As SVM´s tem mostrado um bom desempenho em inumeras aplicações no mundo real, como saúde digital, bioinformatica, mineração de textos e reconhecimento facial. Uma de suas vantagens e que o SVM é muito util quando temos um problema de separação não linear. É possível aplicar novos kernels que permitem felixibilidade para os limites de decisão, obtendo melhores desenpenhos na classificação. Um de seus pontos fracos e que é um pouco lento em comparação com os outros modelos. Escolhi o SVM porque ele permite atravez de uma pequena quantidade de dados trabalhar em modelos assertivos, e prover flexibilidade para suas decisões.
# 
# ####  Árvores de decisão  (DT)
# 
# As árvores de decisão têm sido usadas em áreas importantes de aplicação no campo de pesquisa e pratica médica, alguns exemplos relevantes são observados em dianóstico, cardiologia, psiquiatria entre outros. Os pontos fortes da arvore de decisão são que as relações não lineares não afetam as metricas de desempenho, e possibilitam previsões mais rápidas em comparação com modelos como o SVM. Os pontos fracos são que se tiver uma função de objetivo linear a árvore decisão não é tão indicada, outro ponto fraco e que as árvores de decisão não funcionam bem com limites suaves. Escolho esse modelo porque a árvore de decisão fornece uma metrica de bom desenpenho para dados não lineares.
# 
# ####  Naive Bayes (NB)
# 
# Os métodos Naive Bayes podem ser usados para marcar um e-mail como spam ou não spam, para verificar um texto expressando emoções positivas ou emoções negativas e também em softwares de reconhecimento facial. Os pontos fortes: As vantagens do Naive Bayes é que um classificador NB vai convergir mais rápido do que modelos discriminatórios como a regressão logística, por isso precisamos de menos dados de treinamento. Os pontos fracos consistem na mesma suposição de independência condicional, pode ser uma desvantagem quando não temos ocorrências de um rótulo de classe e um valor de característica em conjunto, o que nos dará uma probabilidade de valor baseado em freqüência zero que afeta qualquer estimativa de probabilidade posterior. Escolhi esse modelo porque o classificar Naive Bayes irá convergir mais rápido do que modelos discriminativos como regressão logística, por isso isso menos dados de treinamento serão necessários.

# ### Configuração
# Execute a célula de código abaixo para inicializar três funções de ajuda que você pode utilizar para treinar e testar os três modelos de aprendizagem supervisionada que você escolheu acima. As funções são as seguintes:
# - `train_classifier` - recebe como parâmetro um classificador e dados de treinamento e ajusta o classificador aos dados.
# - `predict_labels` - recebe como parâmetro um classificador ajustado, atributos e rótulo alvo e faz estimativas utilizando a pontuação do F<sub>1</sub>.
# - `train_predict` - recebe como entrada um classificador, e dados de treinamento e teste, e executa `train_clasifier` e `predict_labels`.
#  - Essa função vai dar a pontuação F<sub>1</sub> tanto para os dados de treinamento como para os de teste, separadamente.

# In[27]:

def train_classifier(clf, X_train, y_train):
    
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    print "O modelo foi treinado em {:.4f} segundos".format(end - start)

    
def predict_labels(clf, features, target):
    
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    print "As previsões foram feitas em {:.4f} segundos.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    
    print "Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train))
    
    train_classifier(clf, X_train, y_train)
    
    print "Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    print "\n"


# ### Implementação: Métricas de Desempenho do Modelo
# Com as funções acima, você vai importar os três modelos de aprendizagem supervisionada de sua escolha e executar a função `train_prediction` para cada um deles. Lembre-se de que você vai precisar treinar e usar cada classificador para três diferentes tamanhos de conjuntos de treinamentos: 100, 200 e 300 pontos. Então você deve ter 9 saídas diferentes abaixo – 3 para cada modelo utilizando cada tamanho de conjunto de treinamento. Na célula de código a seguir, você deve implementar o seguinte:
# - Importe os três modelos de aprendizagem supervisionada que você escolheu na seção anterior.
# - Inicialize os três modelos e armazene eles em `clf_A`, `clf_B` e `clf_C`.
#  - Defina um `random_state` para cada modelo, se a opção existir.
#  - **Nota:** Utilize as configurações padrão para cada modelo – você vai calibrar um modelo específico em uma seção posterior.
# - Crie diferentes tamanhos de conjuntos de treinamento para treinar cada modelo.
#  - *Não embaralhe e distribua novamente os dados! Os novos pontos de treinamento devem ser tirados de `X_train` e `y_train`.*
# - Treine cada modelo com cada tamanho de conjunto de treinamento e faça estimativas com o conjunto de teste (9 vezes no total).  
# **Nota:** Três tabelas são fornecidas depois da célula de código a seguir, nas quais você deve anotar seus resultados.

# In[28]:

from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

clf_A = tree.DecisionTreeClassifier(random_state=42)
clf_B = svm.SVC(random_state=42)
clf_C = GaussianNB()

for size in [100, 200, 300]:
    train_predict(clf_A, X_train[:size], y_train[:size], X_test, y_test)
for size in [100, 200, 300]:
    train_predict(clf_B, X_train[:size], y_train[:size], X_test, y_test)
for size in [100, 200, 300]:
    train_predict(clf_C, X_train[:size], y_train[:size], X_test, y_test)


# ### Resultados Tabulados
# Edite a célula abaixo e veja como a tabela pode ser desenhada em [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). Você deve salvar seus resultados abaixo nas tabelas fornecidas.

# ** Classificador 1 - Árvores de Decisão**  
# 
# | Tamanho do Conjunto de Treinamento | Tempo de Treinamento | Tempo de Estimativa (teste) | Pontuação F1 (treinamento) | Pontuação F1 (teste) |
# | :--------------------------------: |  :------------------: | :-------------------------: | :------------------------: | :------------------: |
# | 100                                |       0.0020               |            0.0000                 |     1.0000                       |        0.6552              | 
# | 200                                |       0.0030        |              0.0000               |        1.0000                    |               0.7500          |   
# | 300                                |          0.0080            |                   0.0010          |        1.0000                    |        0.6613      |
# 
# ** Classificador 2 - Máquinas de Vetores de Suporte**  
# 
# | Tamanho do Conjunto de Treinamento | Tempo de Treinamento | Tempo de Estimativa (teste) | Pontuação F1 (treinamento) | Pontuação F1 (teste) |
# | :--------------------------------: | :------------------: | :-------------------------: | :------------------------: | :------------------: |
# | 100                                |               0.0020       |         0.0020                    |           0.8777                 |          0.7746            |  
# | 200                                |       0.0050        |           0.0040                  |                    0.8679        |                0.7815      |
# | 300                                |             0.0110         |           0.0070                  |              0.8761              |        0.7838      |
# 
# ** Classificador 3 - Naive Bayes**  
# 
# | Tamanho do Conjunto de Treinamento | Tempo de Treinamento | Tempo de Estimativa (teste) | Pontuação F1 (treinamento) | Pontuação F1 (teste) |
# | :--------------------------------: | :------------------: | :-------------------------: | :------------------------: | :------------------: |
# | 100                                |       0.0020               |               0.0010              |          0.8467                  |            0.8029          |
# | 200                                |           0.0010           |               0.0010              |      0.8406                      |            0.7244          |
# | 300                                |       0.0010               |                  0.0000           |         0.8038                   |          0.7634            |

# ## Escolhendo o Melhor Modelo
# Nesta seção final, você irá escolher dos três modelos de aprendizagem supervisionada o *melhor* para utilizar os dados dos estudantes. Você então executará um busca em matriz otimizada para o modelo em todo o conjunto de treinamento (`X_train` e `y_train`) ao calibrar pelo menos um parâmetro, melhorando em comparação a pontuação F<sub>1</sub> do modelo não calibrado. 

# ### Questão 3 - Escolhendo o Melhor Modelo
# *Baseando-se nos experimentos que você executou até agora, explique em um ou dois parágrafos ao conselho de supervisores qual modelo que você escolheu como o melhor. Qual modelo é o mais apropriado baseado nos dados disponíveis, recursos limitados, custo e desempenho?*

# **Resposta: ** A partir dos resultados avaliados e da média de seus resultados de métricas, o metodo que apresentou a taxa de predição mais assertiva no conjunto de teste foi o o das Máquinas de Vetores de Suporte, com sua pontuação F1 de 0,7799, enquanto Árvores de Decisão com 0,6888 e o Naive Bayes em 0,7635. O desenpenho para treino e predições é mais lento em comparação com outros, o modelo SVM se mostra o mais assertivo para esse exemplo. Existindo uma reavaliação de recursos, afim de priorizar os recursos mais importantes para revisão, é possivel otimizar o desempenho do modelo em questão. Atravez das métricas de desenpenho percebe-se que mesmo em 200 amostras de dados, o modelo SVM retorna o melhor desenpenho em comparação com os outros, 0,7815. Ficou claro que o modelo SVM é mais assetivo quando falamos de um número limitado de dados dispóniveis.

# ### Questão 4 – O Modelo para um Leigo
# *Em um ou dois parágrafos, explique para o conselho de supervisores, utilizando termos leigos, como o modelo final escolhido deve trabalhar. Tenha certeza que você esteja descrevendo as melhores qualidades do modelo, por exemplo, como o modelo é treinado e como ele faz uma estimativa. Evite jargões técnicos ou matemáticos, como descrever equações ou discutir a implementação do algoritmo.*

# **Resposta: ** O modelo que eu escolhi como melhor para classificar os alunos suceptíveis de sofrer intervenções ou não é o das Máquinas de Vetores de Suporte (SVM). Esse modelo utiliza os dados de alunos anteriores para fazer previsões sobre dados dos novos alunos. O modelo basicamente ira ser ensinado atravez dos dados obtidos com os alunos anteriores, criando uma margem entre os alunos que passaram de ano e aqueles que não passaram. Após haver uma margem estabelecida, são inseridos os modelos para os novos dados obtidos com os novos alunos. Dessa forma esse modelo será capaz de prever atravez dos dados obtidos se um estudante vai passar de ano ou não. Um exemplo pratico está nesse modelo que estou usando para minha startup NeuroAppp, aonde fazemos um radar de dificuldades de aprendizagem, e preciso de um modelo, comparando alunos anteriores com as dificuldades e aplicando os modelos para os novos alunos. Percebi que o metodo das Máquinas de Vetores de Suporte será essencial para a construção do nosso radar de dificuldades de aprendizagem.

# ### Implementação: Calibrando o Modelo
# Calibre o modelo escolhido. Utilize busca em matriz (`GridSearchCV`) com, pelo menos, um parâmetro importante calibrado com, pelo menos, 3 valores diferentes. Você vai precisar utilizar todo o conjunto de treinamento para isso. Na célula de código abaixo, você deve implementar o seguinte:
# - Importe [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) e [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Crie o dicionário de parâmetros que você deseja calibrar para o modelo escolhido.
#  - Examplo: `parameters = {'parameter' : [list of values]}`.
# - Inicialize o classificador que você escolheu e armazene-o em `clf`.
# - Crie a função de pontuação F<sub>1</sub> utilizando `make_scorer` e armazene-o em `f1_scorer`.
#  - Estabeleça o parâmetro `pos_label` para o valor correto!
# - Execute uma busca em matriz no classificador `clf` utilizando o `f1_scorer` como método de pontuação e armazene-o em `grid_obj`.
# - Treine o objeto de busca em matriz com os dados de treinamento (`X_train`, `y_train`) e armazene-o em `grid_obj`.

# In[31]:

from sklearn import svm, grid_search, datasets
from sklearn.metrics import make_scorer
from sklearn.svm import SVC

parameters = [{'C': [1, 10, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]

clf = SVC()

f1_scorer = make_scorer(f1_score, pos_label='yes')

grid_obj = grid_search.GridSearchCV(clf,param_grid = parameters, scoring = f1_scorer)

grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_

print "O modelo calibrado tem F1 de {:.4f} no conjunto de treinamento.".format(predict_labels(clf, X_train, y_train))
print "O modelo calibrado tem F1 de {:.4f} no conjunto de teste.".format(predict_labels(clf, X_test, y_test))


# ### Questão 5 - Pontuação F<sub>1</sub> Final
# *Qual é a pontuação F<sub>1</sub> do modelo final para treinamento e teste? Como ele se compara ao modelo que não foi calibrado?*

# **Resposta: ** A pontuação F1 do modelo final é 0.8423 para conjunto de treino, e 0.7838 para conjunto de teste. Em comparação com o modelo que não foi calibrado esse é mais assertivo, afinal no modelo que não foi calibrado temos 0,7799 para conjunto de testes e 0,8739. É uma melhora pequena porêm já mostra que os parametros padrões foram bem escolhidos.

# > **Nota**: Uma vez que você completou todas as implementações de código e respondeu todas as questões acima com êxito, você pode finalizar seu trabalho exportando o iPython Nothebook como um document HTML. Você pode fazer isso utilizando o menu acima e navegando para  
# **File -> Download as -> HTML (.html)**. Inclua a documentação final junto com o notebook para o envio do seu projeto.

# In[ ]:



