# NOTAS GERAIS

1. Vou precisar aprender mais sobre a transformada de laplace.

## Reunião 18/03

1. Separar as quatro componentes do sinal $S_0$, $\omega$, $\alpha$, $\phi$ dos valores calculados de $z_i$ e $R_i$. Tais componentes são da equação. **COMPLETO**

    $$S = S_0 e^{-(\alpha - i\omega)t + \phi}$$
    
    Adicionar fase $\phi$ para esses testes.

2. Validar a tabela anterior. **COMPLETO**
3. Implementar visualização de comparação entre o sinal simulado, reconstruído e o resíduo (estimado - simulado). **COMPLETO**
4. Testar variação do $L$ de $700$, $800$, $900$ e $1000$ pontos (valor de corte do svd mais baixo possível). Rodar isso $K$ vezes para ter certeza que os valores da tabela anteriormente implementada não sofrem intereferência de $L$. Rodar primeiro para um K baixo, se houver alguma interferência sequer, rodar para $K$ maiores. **INCOMPLETO**
5. Testar para a variação do corte no SVD. **INCOMPLETO**
