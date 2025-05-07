# NOTAS GERAIS

1. Vou precisar aprender mais sobre a transformada de laplace.

## Reunião 18/03

1. Separar as quatro componentes do sinal $S_0$, $\omega$, $\alpha$, $\phi$ dos valores calculados de $z_i$ e $R_i$. Tais componentes são da equação. **COMPLETO**

    $$S = S_0 e^{-(\alpha - i\omega)t + \phi}$$
    
    Adicionar fase $\phi$ para esses testes.

2. Validar a tabela anterior. **COMPLETO**
3. Implementar visualização de comparação entre o sinal simulado, reconstruído e o resíduo (estimado - simulado). **COMPLETO**
4. Testar variação do $L$ de $700$, $800$, $900$ e $1000$ pontos (valor de corte do svd mais baixo possível). Rodar isso $K$ vezes para ter certeza que os valores da tabela anteriormente implementada não sofrem intereferência de $L$. Rodar primeiro para um K baixo, se houver alguma interferência sequer, rodar para $K$ maiores. **COMPLETO** [examples/MPM/L_testing.ipynb](examples/MPM/L_testing.ipynb)
    - Fazer o cálculo do RMSE para o sinal limpo também.
5. Testar para a variação do corte no SVD. **COMPLETO** [examples/MPM/svd_testing.ipynb](examples/MPM/svd_testing.ipynb)
6. Testar o L e o corte do SVD em um sinal com ruído. **INCOMPLETO**
7. Simular um sinal com apenas um pico (nacetylaspartato), rodar o MPM e analisar quais componentes resultaram do algoritmo (omega, amplitude, fase, etc). Gerar o sinal com determinada fase a analisar de novo. Olhar pra componente de interesse do array e as demais (como os valores das demais que em teoria seriam zero se comportam). **INCOMPLETO**
8. Adicionar ruído baixo no sinal acima e acompanhar a evolução das componentes conforme o ruído aumenta. Aumentar o nível de ruído para entender quantas componentes de valores significativos aparecem. Comparar o sinal original e reconstruído, que apesar de iguais, já não são representados pelos parametros originais. **COMPLETO** [examples/MPM/single_peak_noise.ipynb](examples/MPM/single_peak_noise.ipynb)
9. *RESPONDIDO: (**não faz diferença**)* Constatei que a função geradora das frequencias está gerando com valor positivo de frequencia e não negativo. Sei que isso altera apenas a fase, mas devo corrigir? Qual é o correto? *Se alterar na função original, deve-se alterar outras funções que levam isso em consideração, incluindo a que calcula as variáveis a partir de z e r*
10. Filtrar as componentes do sinal e plotar vários gráficos dessas várias componentes geradas individualmente sobrepostas, e do lado plotar outro gráfico com a soma individual dessas componentes. **COMPLETO** [examples/MPM/spec_components.ipynb](examples/MPM/spec_components.ipynb)
11. Mudar o critério de numero de componentes no programa `spec_components.ipynb`, o `n_comp`, para filtrar apenas o numero de componentes que tem pico de valor até tantos por cento menor que o maior pico (componentes abaixo de, por exemplo, 1% do valor do maior pico não serão analisadas). **COMPLETO**
12. A partir da mudança acima, fazer um gráfico que relacione o *número de picos*, o *snr*, e talvez um teceiro eixo, o do *RMSE* entre a amostra original e a reconstruída com apenas `n_comp` componentes. Caso a visualização 3D não funcione, fazer um gráfico separado. Analisar o RMSE na área do pico e no sinal inteiro, ou seja, terá duas informações de RMSE. **INCOMPLETO**
13. Reduzir o intervalo de SNR para [500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000]. **COMPLETO**
14. Fazer uma análise visual para cada gráfico com SNR gerado e analisar para ver como se comporta o resíduo (original - reconstruído) na região do pico. Verificar se dentro dos picos individuais o pico original existe, e se não, se há vários picos individuais que reconstroem o pico original. **INCOMPLETO**
