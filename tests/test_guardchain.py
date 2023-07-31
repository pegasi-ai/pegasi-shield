import unittest
from guardrail.guardchain import Chain, BufferMemory, BaseLanguageModel, ChatAgent

class TestGuardChainInit(unittest.TestCase):
    test_prompt="Write me a poem about AI"

    def test_guardchain_init(self):
        llm = BaseLanguageModel(model_name="EleutherAI/pythia-410m-deduped-v0")
        memory = BufferMemory()
        agent = ChatAgent.from_llm_and_tools(llm=llm)
        chain = Chain(agent=agent, memory=memory)
        output = chain.run(self.test_prompt)['message']
        print(output)
        self.assertGreater(len(output), 1)

unittest.main()
