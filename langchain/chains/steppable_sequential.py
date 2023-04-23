"""Chain pipeline where the outputs of one step feed directly into next, but can be paused and stepped."""
from typing import Dict
import asyncio

from langchain.chains.sequential import SequentialChain


class SteppableSequentialChain(SequentialChain):
    """Chain where the outputs of one chain feed directly into next."""

    paused: bool = True
    known_values = {}
    current_chain = 0
    is_running = asyncio.Event()
    # pause_scheduled = False

    class NoMoreChainsError(Exception):
        pass

    class ChainCompletedError(Exception):
        pass

    async def pause(self) -> None:
        print("pausing func start")
        if self.current_chain == len(self.chains):
            return
        # self.pause_scheduled = True
        self.is_running.clear()
        self.paused = True
        print("pausing func end")

    async def play(self) -> None:
        print("playing func start")
        if self.current_chain == len(self.chains):
            return
        # self.pause_scheduled = True
        self.is_running.set()
        self.paused = False
        print("playing func end")

    async def stepper(self) -> Dict[str, str]:
        while self.current_chain < len(self.chains):
            print(f"is running set: {self.is_running.is_set}")
            await self.is_running.wait()
            self.step()
            # if self.pause_scheduled:
            #     self.paused = True
            #     self.pause_event.set()
            #     self.pause_scheduled = False
            print(f"completed subchain step {self.current_chain - 1}")
            print(f"waiting 0.5 secs before next step")
            await asyncio.sleep(0.5)

        print("chain completed")
        print(f"output_variables: {self.output_variables}")
        print(f"known_values: {self.known_values}")
        for k in self.output_variables:
            print(f"k: {k}")
            print(f"known_values[k]: {self.known_values[k]}")
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
        print("kickoff chain")
        await self.play()
        return await self.stepper()

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        raise NotImplementedError(
            "Sync call not supported for this chain type - use async."
        )
