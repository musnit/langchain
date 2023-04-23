"""Integration test for self ask with search."""

import asyncio
from typing import Dict, List

import pytest
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain
from langchain.chains.steppable_sequential import SteppableSequentialChain
from langchain.llms.openai import OpenAI


class ConcatenateCoolChain(Chain):
    chain_1: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys)
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ["cool_name"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        return {"cool_name": output_1 + " cool!"}


@pytest.mark.asyncio
async def test_steppable() -> None:
    llm = OpenAI(temperature=1)

    prompt_1 = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_1)
    coolname_chain = ConcatenateCoolChain(chain_1=name_chain, output_key="cool_name")

    # prompt_2 = PromptTemplate(
    #     input_variables=["product"],
    #     template="What is a good slogan for a company that makes {product}?",
    # )
    # chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    cool_template = """Do you think this name is cool? The name is {cool_name}"""
    prompt_template = PromptTemplate(
        input_variables=["cool_name"], template=cool_template
    )
    coolcheck_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SteppableSequentialChain(
        chains=[coolname_chain, coolcheck_chain],
        input_variables=["product"],
        output_variables=["text"],
        verbose=True,
    )

    result_promise = asyncio.create_task(overall_chain.arun({"product": "Eggs"}))

    print("waiting for 0.5 seconds...")

    await asyncio.sleep(0.5)

    print("pausing chain...")
    await overall_chain.pause()
    print("chain paused...")

    print("waiting 5 seconds...")
    await asyncio.sleep(5)
    print("resuming chain...")
    asyncio.create_task(overall_chain.play())

    print("waiting 2 seconds...")
    await asyncio.sleep(2)

    print("waiting for result...")
    result = await result_promise
    print(f"Final result: {result}. End.")

    # assert result["cool_output"] == "cool!"
