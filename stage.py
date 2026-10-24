from abc import ABC, abstractmethod


class Stage(ABC):
    """Classe base para todos os estágios do pipeline SimEDaPE.

    Cada estágio recebe o contexto compartilhado, executa sua lógica e
    pode modificar o contexto para os estágios seguintes.

    Subclasses devem definir:
    - `name` (str): identificador único do estágio
    - `execute(ctx)`: implementação do estágio
    """

    name: str = "stage"

    @abstractmethod
    def execute(self, ctx: "PipelineContext") -> None:  # noqa: F821
        ...

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<Stage: {self.name}>"
