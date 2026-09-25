# 🐱 Go-Cat

Go-Cat é um projeto experimental de **vida artificial e aprendizado por reforço**, criado para explorar, de forma incremental, como um agente pode aprender a interagir com um ambiente, tomar decisões e adaptar seu comportamento a partir das consequências de suas ações.

O projeto começa propositalmente simples: um gato se movimenta em um mundo baseado em uma grade, procura uma fonte de comida e aprende, através de **Q-Learning**, quais ações são mais vantajosas.

A ideia não é criar uma IA pronta ou um jogo tradicional, mas construir um pequeno laboratório onde diferentes conceitos de inteligência artificial possam ser implementados e observados.

---

## 🎯 Objetivo

O objetivo do Go-Cat é evoluir gradualmente de um experimento simples de aprendizado por reforço para um ambiente de **vida artificial**, no qual agentes possam desenvolver comportamentos cada vez mais complexos.

A evolução do projeto será incremental.

O agente começa sabendo apenas:

* quais ações pode executar;
* como interagir com o ambiente;
* quais recompensas recebe.

Todo o comportamento aprendido deve surgir através da interação com o ambiente e das experiências acumuladas pelo agente.

---

## 🧠 Conceito atual

Atualmente existe um agente representado por um gato em um ambiente de grade.

O ambiente possui:

```text
10 × 10
```

O gato começa em:

```text
[0, 0]
```

A comida começa em:

```text
[7, 7]
```

O agente possui quatro ações possíveis:

```text
up
down
left
right
```

Cada movimento normalmente produz uma recompensa de:

```text
-1
```

Ao encontrar a comida:

```text
+10
```

O episódio termina quando o gato chega até a comida.

---

## 🤖 Aprendizado por reforço

O algoritmo utilizado atualmente é **Q-Learning**.

O agente mantém uma Q-table que associa:

```text
estado → ação → valor
```

Por exemplo:

```text
(0, 0)

up     → 0.0
down   → 0.0
left   → 0.0
right  → 0.0
```

Conforme o agente interage com o ambiente, esses valores são atualizados.

Uma ação que produz uma experiência ruim tende a receber um valor menor.

Uma ação que conduz a uma recompensa positiva tende a receber um valor maior.

Com o treinamento, o agente passa a utilizar essas informações para tomar decisões.

---

## 🔍 Exploração e aproveitamento

O agente utiliza uma estratégia **epsilon-greedy**.

Durante o início do treinamento, existe uma alta taxa de exploração:

```text
exploration_rate = 1.0
```

Isso significa que o agente frequentemente escolhe ações aleatórias.

A taxa de exploração diminui gradualmente durante o treinamento:

```text
1.000
 ↓
0.606
 ↓
0.367
 ↓
...
 ↓
0.010
```

Com isso, o agente começa explorando o ambiente e posteriormente passa a aproveitar mais aquilo que aprendeu.

---

## 📈 Resultado atual

Depois de aproximadamente 1000 episódios de treinamento, o agente consegue encontrar a comida consistentemente utilizando um caminho de tamanho mínimo.

A distância entre:

```text
[0, 0]
```

e:

```text
[7, 7]
```

é de:

```text
14 movimentos
```

(7 movimentos horizontais + 7 movimentos verticais).

O agente consegue atingir esse resultado sem receber uma rota pré-programada.

Ele aprende os valores das ações através das experiências obtidas durante o treinamento.

---

## 🏗️ Arquitetura

O projeto utiliza uma estrutura baseada em pacotes Python para manter os diferentes componentes separados.

```text
go-cat/
│
├── README.md
├── pyproject.toml
│
├── scripts/
│   └── train.py
│
├── src/
│   └── ai_cat/
│       │
│       ├── agents/
│       │   ├── agent.py
│       │   └── __init__.py
│       │
│       ├── environments/
│       │   ├── grid_world.py
│       │   └── __init__.py
│       │
│       ├── experiments/
│       │   ├── experiment.py
│       │   └── __init__.py
│       │
│       ├── learning/
│       │   ├── q_learning.py
│       │   └── __init__.py
│       │
│       └── simulation/
│           └── __init__.py
│
└── tests/
    ├── test_agent.py
    ├── test_environment.py
    ├── test_exploration.py
    ├── test_learning.py
    └── test_trained_agent.py
```

### Agent

Responsável pelo agente e pelo seu estado dentro do ambiente.

Atualmente possui:

* posição;
* conjunto de ações;
* mecanismo de aprendizado.

O agente não implementa diretamente o algoritmo de Q-Learning.

---

### QLearning

Localizado em:

```text
src/ai_cat/learning/q_learning.py
```

Responsável pelo algoritmo de aprendizado.

Atualmente controla:

* Q-table;
* learning rate;
* discount factor;
* exploration rate;
* exploration decay;
* escolha de ações;
* atualização dos valores;
* redução da exploração.

Essa separação permite que diferentes algoritmos de aprendizado sejam adicionados futuramente sem concentrar toda a lógica dentro do agente.

---

### Environment

Localizado atualmente em:

```text
src/ai_cat/environments/grid_world.py
```

Responsável pelo mundo onde o agente existe.

Atualmente controla:

* tamanho da grade;
* posição do gato;
* posição da comida;
* movimentação;
* recompensas;
* encerramento do episódio.

---

### Experiments

Contém experimentos utilizados para observar comportamentos específicos do sistema.

---

### Scripts

Contém scripts executáveis para tarefas relacionadas ao projeto.

Atualmente:

```text
scripts/train.py
```

é utilizado para realizar o treinamento do agente.

---

## 🧪 Testes

O projeto utiliza **pytest** para validar seu comportamento.

Atualmente existem 11 testes automatizados cobrindo:

* criação do agente;
* criação da Q-table;
* estado inicial do ambiente;
* reset do ambiente;
* movimentação;
* chegada até a comida;
* exploração;
* aprendizado;
* atualização dos valores da Q-table;
* treinamento completo;
* capacidade do agente treinado de encontrar a comida.

Os testes podem ser executados com:

```bash
pytest
```

Resultado atual:

```text
11 passed
```

A suíte de testes é importante porque o projeto será evoluído constantemente. Cada nova mecânica deve poder ser adicionada sem quebrar os comportamentos já existentes.

---

## ⚙️ Tecnologias

Atualmente o projeto utiliza:

* Python 3.10+
* NumPy
* Matplotlib
* pytest
* setuptools
* Q-Learning

---

## 🚀 Instalação

Clone o projeto e entre no diretório:

```bash
git clone <repository-url>
cd go-cat
```

Crie o ambiente virtual:

```bash
python3 -m venv .venv
```

Ative:

```bash
source .venv/bin/activate
```

Instale o projeto:

```bash
python -m pip install --upgrade pip setuptools
python -m pip install numpy matplotlib pytest
python -m pip install -e .
```

---

## ▶️ Treinamento

Com o ambiente virtual ativado:

```bash
python scripts/train.py
```

O treinamento executa uma série de episódios nos quais o agente interage com o ambiente e atualiza sua Q-table.

---

## 🧪 Executando os testes

Para executar toda a suíte:

```bash
pytest
```

Para executar um arquivo específico:

```bash
pytest tests/test_agent.py
```

Por exemplo:

```bash
pytest tests/test_trained_agent.py
```

---

## 🔬 Filosofia do projeto

O Go-Cat será desenvolvido de forma incremental.

A intenção é evitar começar com um sistema complexo de inteligência artificial e tentar fazer tudo funcionar de uma vez.

Em vez disso:

```text
agente simples
      ↓
aprendizado
      ↓
ambiente mais complexo
      ↓
novos desafios
      ↓
novos comportamentos
      ↓
novas capacidades
```

Cada etapa deve introduzir uma nova capacidade e permitir observar como ela altera o comportamento do agente.

---

## 🗺️ Possíveis evoluções

O projeto foi estruturado para permitir a introdução gradual de novas mecânicas.

Algumas possibilidades futuras:

### Ambiente

* obstáculos;
* diferentes tipos de terreno;
* múltiplos alimentos;
* água;
* áreas perigosas;
* mudanças no ambiente;
* ambientes maiores;
* múltiplos ambientes.

### Necessidades do agente

* fome;
* sede;
* energia;
* descanso;
* segurança;
* temperatura;
* necessidades simultâneas.

### Comportamento

* memória;
* curiosidade;
* exploração;
* preferência por locais;
* reconhecimento de padrões;
* tomada de decisão baseada em múltiplas necessidades.

### Inteligência

* diferentes algoritmos de aprendizado;
* redes neurais;
* Deep Q-Learning;
* aprendizado por transferência;
* evolução de agentes;
* algoritmos genéticos.

### Vida artificial

Em estágios mais avançados, o projeto poderá explorar conceitos como:

* múltiplos agentes;
* competição;
* cooperação;
* comunicação;
* reprodução;
* gerações;
* características herdadas;
* adaptação ao ambiente;
* surgimento de comportamentos não programados diretamente.

Essas possibilidades não fazem parte do sistema atual. Elas representam direções experimentais para futuras etapas do projeto.

---

## 📌 Estado atual

O projeto encontra-se em uma fase inicial de desenvolvimento.

Atualmente:

```text
✓ Estrutura de projeto baseada em pacote Python
✓ Ambiente virtual
✓ Agente
✓ Grid World
✓ Q-table
✓ Q-Learning
✓ Exploração epsilon-greedy
✓ Treinamento
✓ Agente treinado encontrando comida
✓ Separação entre Agent e algoritmo de aprendizado
✓ Testes automatizados
✓ 11 testes passando
```

O próximo estágio será aumentar gradualmente a complexidade do ambiente e observar como o comportamento aprendido pelo agente se modifica.

---

## 🐱 Projeto experimental

Go-Cat é, acima de tudo, um laboratório para experimentar ideias relacionadas a **aprendizado por reforço, agentes autônomos e vida artificial**.

A complexidade deve surgir aos poucos, permitindo observar não apenas se o agente consegue resolver um problema, mas **como seu comportamento muda quando novas condições são introduzidas no mundo em que ele vive**.
