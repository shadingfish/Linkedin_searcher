from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

# from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile

# from third_parties.twitter import scrape_user_tweets
from output_parsers import person_intel_parser, PersonIntel


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    summary_template = """
         given the Linkedin information {linkedin_information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
         3. A topic that may interest them
         4. 2 creative Ice breakers to open a conversation with them
         \n{format_instruction}
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information"],
        template=summary_template,
        partial_variables={
            "format_instruction": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    chain_result = chain.run(linkedin_information=linkedin_data)
    print(chain_result)
    # print(result)
    return person_intel_parser.parse(chain_result), linkedin_data.get("profile_pic_url")
    # tweets = scrape_user_tweets(username="@elonmusk", num_tweets=100)
    # print(tweets)


if __name__ == "__main__":
    print("Hello LangChain!")
    result = ice_break(name="Harrison Chase")
    print(result)
