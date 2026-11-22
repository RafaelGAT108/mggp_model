
from collections import Counter
from deap import gp
import numpy as np

RHO_OPS = 1.0               # principal: uma multiplicação custa o mesmo que uma adição
RHO_KARATSUBA = 729 / 64    # secundário (~11,4): peso da complexidade Green-Box original

def gene_monomial(tree: gp.PrimitiveTree) -> tuple:
    """Converto um gene (árvore) no seu monômio: tupla ordenada de ((variável, atraso), expoente)."""
    def parse(i: int, shift: int):
        node = tree[i]
        if isinstance(node, gp.Terminal):                 # folha: uma variável (y1, u1, ...)
            return Counter({(str(node.value), shift): 1}), i + 1
        name = node.name
        if name.startswith("q") and node.arity == 1:      # q<j>: soma j ao atraso de tudo abaixo
            return parse(i + 1, shift + int(name[1:]))
        if name == "mul":                                 # mul: junta os fatores dos dois lados
            left, j = parse(i + 1, shift)
            right, k = parse(j, shift)
            return left + right, k
        raise ValueError(f"o custo só aceita 'mul' e atrasos 'q', encontrei '{name}'")

    factors, _ = parse(0, 0)
    return tuple(sorted(factors.items()))


def canonical_terms(individual) -> frozenset:
    """Conjunto dos termos DIFERENTES do indivíduo.

    Se dois genes dão o mesmo termo, o modelo executado tem esse termo uma vez só (os
    coeficientes dos dois se somam). Por isso o custo é calculado sobre este conjunto,
    e não sobre a lista de genes.
    """
    return frozenset(gene_monomial(g) for g in individual)


def arithmetic_cost(individual, rho: float = RHO_OPS) -> float:
    """Custo aritmético C = a + rho * b do polinômio que realmente é executado.

    O modelo é  y(k) = theta_0 + theta_1 * phi_1 + ... + theta_{p-1} * phi_{p-1}:
    p termos ao todo, CONTANDO o termo constante theta_0.
      a = p - 1  adições: somar p parcelas exige p - 1 somas (2 parcelas -> 1 soma, 3 -> 2, ...).
                 O constante conta como termo; só não gera uma soma a mais, porque é a 1ª parcela.
      b = d_1 + ... + d_{p-1}  multiplicações: o termo theta_i * phi_i de grau d_i (número de
                 fatores de u e y) faz d_i multiplicações; o constante não multiplica nada.
    Exemplo: theta_0 + theta_1*u(k) + theta_2*y(k-1)*u(k)  ->  p = 3, a = 2, b = 1 + 2 = 3,
    C = 5 (com rho = 1).
    """
    terms = canonical_terms(individual)    # termos não constantes, sem repetição (são p - 1)
    a = len(terms)                         # = p - 1 adições
    b = sum(sum(power for _, power in term) for term in terms)
    return a + rho * b


def make_cost(individual, rho: float = RHO_OPS):
    """Devolvo a função de custo com o rho fixo, no formato que o pacote espera (new_evaluation)."""
    def cost(individual) -> float:
        return arithmetic_cost(individual, rho)
    return cost(individual)


def count_times(individual, times_weight: float = 0.66) -> float:
    """Complexidade do tutorial do Rafael (número de 'mul' x 0,66). Mantive só para comparação:
    ela conta genes repetidos e não conta as multiplicações pelos coeficientes."""
    return sum(str(g).count("mul") * times_weight for g in individual)


def finite(ind) -> bool:
    """Indivíduo avaliado com erro e custo finitos (modelos que divergem ficam com infinito)."""
    return ind.fitness.valid and all(np.isfinite(ind.fitness.values))