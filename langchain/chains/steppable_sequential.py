"""Chain pipeline where the outputs of one step feed directly into next, but can be paused and stepped."""
from typing import Any, Dict
import asyncio

from langchain.chains.sequential import SequentialChain


class SteppableSequentialChain(SequentialChain):
    """Chain where the outputs of one chain feed directly into next."""

    paused: bool = True
    known_values = {}
    current_chain = 0
    is_running = asyncio.Event()

    @property
    def _chain_type(self) -> str:
        return "steppable_sequential_chain"

    class NoMoreChainsError(Exception):
        pass

    class ChainCompletedError(Exception):
        pass

    async def pause(self) -> None:
        if self.current_chain == len(self.chains):
            return
        self.is_running.clear()
        self.paused = True

    async def play(self) -> None:
        if self.current_chain == len(self.chains):
            return
        self.is_running.set()
        self.paused = False

    async def stepper(self) -> Dict[str, str]:
        while self.current_chain < len(self.chains):
            await self.is_running.wait()
            self.step()
            await asyncio.sleep(0.5)

        return {k: self.known_values[k] for k in self.output_variables}

    def step(self) -> None:
        if self.current_chain < len(self.chains):
            print(f"running subchain step {self.current_chain}")
            chain = self.chains[self.current_chain]
            outputs = chain(self.known_values, return_only_outputs=True)
            self.known_values.update(outputs)
            self.current_chain += 1
        else:
            raise self.NoMoreChainsError("No more chains to execute.")

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        self.known_values = inputs.copy()
        self.current_chain = 0
        await self.play()
        return await self.stepper()

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        raise NotImplementedError(
            "Sync call not supported for this chain type - use async."
        )

    def dict(self, **kwargs: Any):
        kwargs.setdefault("exclude", set()).add("is_running")
        return super().dict(**kwargs)
