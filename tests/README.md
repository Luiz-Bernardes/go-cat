# Testes e Experimentos

Os testes deste diretório não representam apenas testes
unitários tradicionais. Alguns deles são experimentos
utilizados para investigar o comportamento do agente,
da percepção, da memória e do aprendizado.

## Estrutura

### agent/

Testes relacionados ao comportamento básico do Agent.

### environment/

Testes relacionados ao ambiente e suas regras.

### learning/

Testes do mecanismo de aprendizado e exploração.

### memory/

Testes da memória temporal do agente.

### perception/

Experimentos relacionados à percepção e suas limitações.

### state/

Experimentos relacionados à transformação da memória
em estados utilizados pelo aprendizado.

## Linha de evolução

A investigação atual segue aproximadamente esta sequência:

Environment
↓
Perception
↓
Perception Learning
↓
Generalization
↓
Perception Limitation
↓
Memory
↓
Memory Length
↓
Memory State
↓
Memory State Learning