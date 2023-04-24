"""Integration test for self ask with search."""

import asyncio

import pytest
from langchain import LLMChain, PromptTemplate
from langchain.chains.steppable_sequential import SteppableSequentialChain
from langchain.llms.openai import OpenAI
from langchain.chains import load_chain


@pytest.mark.asyncio
async def test_steppable() -> None:
    llm = OpenAI(temperature=1)

    name_prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key="name")

    cool_template = """Do you think this name is cool? The name is {name}. Please make sure to repeat the name in your answer."""
    prompt_template = PromptTemplate(input_variables=["name"], template=cool_template)
    coolcheck_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="is_cool")

    overall_chain = SteppableSequentialChain(
        chains=[name_chain, coolcheck_chain],
        input_variables=["product"],
        output_variables=["is_cool"],
        verbose=True,
    )

    asyncio.create_task(overall_chain.arun({"product": "Eggs"}))

    print("waiting for 0.5 seconds...")

    await asyncio.sleep(0.5)

    print("pausing chain...")
    await overall_chain.pause()
    print("chain paused...")

    print("saving chain..")
    save_path = "saved-chain.yaml"
    overall_chain.save(file_path=save_path)
    del overall_chain

    restored_chain = load_chain(save_path)
    print(f"Deserialized chain: {restored_chain}")

    print(f"Resuming chain")

    await restored_chain.play()

    result_promise = asyncio.create_task(restored_chain.stepper())

    print("waiting 1 seconds...")
    await asyncio.sleep(1)

    print("waiting for result...")
    result = await result_promise
    print(f"Final result: {result}. End.")

    # assert result["cool_output"] == "cool!"
