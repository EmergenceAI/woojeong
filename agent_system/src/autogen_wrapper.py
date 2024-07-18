import autogen
import os
import logging
import tempfile
import json
import inspect
from prompts import LLM_PROMPTS
from dotenv import load_dotenv
from agent_system.src.utils import terminate, convert_to_openai_tool_schema, convert_api_to_function
from agent_system.src.tool_simulator import APISimulator
        

class AutogenWrapper:
    """
    A wrapper class for interacting with the Autogen library.

    Args:
        max_chat_round (int): The maximum number of chat rounds.

    Attributes:
        number_of_rounds (int): The maximum number of chat rounds.
        agents_map (dict): A dictionary of the agents that are instantiated in this autogen instance.

    """

    def __init__(self, max_chat_round: int = 1000):
        self.max_chat_round = max_chat_round

        self.agents_map: dict[str, autogen.AssistantAgent | autogen.UserProxyAgent] = None
        self.llm_config: dict[dict[str, str]] | None = None
    
    def create(self, agents_needed: list[str] = ["user", "orchestrator", "tool_executor", "tool_execution_manager"]):
        """
        Create an instance of AutogenWrapper.

        Args:
            agents_needed (list[str], optional): The list of agents needed. Defaults to ["user", "orchestrator", "tool_executor", "tool_execution_manager"].
            max_chat_round (int, optional): The maximum number of chat rounds. Defaults to 50.

        Returns:
            AutogenWrapper: An instance of AutogenWrapper.

        """
        # === Configure the environment variables ===
        print(f">>> Creating AutogenWrapper with {agents_needed} and {self.max_chat_round} rounds.")
        # # Create an instance of cls
        # self = cls(max_chat_round, use_groupchat)
        load_dotenv()
        os.environ["AUTOGEN_USE_DOCKER"] = "False"

        autogen_model_name = os.getenv("AUTOGEN_MODEL_NAME")
        if not autogen_model_name:
            autogen_model_name = "gpt-4-turbo"
            logging.warning(f"Cannot find AUTOGEN_MODEL_NAME in the environment variables, setting it to default {autogen_model_name}.")

        autogen_model_api_key = os.getenv("OPENAI_API_KEY")
        if autogen_model_api_key is None:
            raise ValueError("You need to set OPENAI_API_KEY in the .env file.")
        else:
            logging.info(f"Using model {autogen_model_name} for AutoGen from the environment variables.")
        model_info = {'model': autogen_model_name, 'api_key': autogen_model_api_key}

        env_var: list[dict[str, str]] = [model_info]
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
            json.dump(env_var, temp)
            temp_file_path = temp.name
        self.llm_config = {
            "config_list": autogen.config_list_from_json(env_or_file=temp_file_path, filter_dict={"model": {autogen_model_name}}),
            "cache_seed": None,
            "temperature": 0.0,
        }
        
        # === Initialize the agents ===
        self.agents_map = self._initialize_agents(agents_needed)
        print(f">>> Agents initialized: {self.agents_map}")

        # === Register nested chat ===
        if len(self.agents_map) == 5:
            # default setting
            # ['user', 'orchestrator', 'tool_executor', 'tool_execution_manager', 'tool_groupchat_manager']
            def tool_execution_manager_message(recipient, messages, sender, config):
                return f"Please select the most relevant tools and execute them: {recipient.chat_messages_for_summary(sender)[-1]['content']}"
            nested_chat_queue = [
                {"recipient": self.agents_map["tool_groupchat_manager"], "message": tool_execution_manager_message, "summary_method": "reflection_with_llm"},
            ]
        elif "planner" in self.agents_map.keys():
            def planner_message(recipient, messages, sender, config):
                return f"Come up with a step-by-step plan to solve the given user query. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"
            
            nested_chat_queue = [
                {"recipient": self.agents_map["planner"], "message": planner_message, "summary_method": "last_msg", "max_turns": 1},
                {"recipient": self.agents_map["tool_groupchat_manager"], "summary_method": "reflection_with_llm"},
            ]
        elif "tool_retriever" in self.agents_map.keys():
            raise NotImplementedError("Tool retriever agent is not implemented yet.")
        else:
            raise ValueError("Unknown agents configuration.")

        self.agents_map["orchestrator"].register_nested_chats(
            nested_chat_queue,
            trigger=self.agents_map["user"],
        )
    
    def _initialize_agents(self, agents_needed: list[str]):
        """
        Instantiate all agents with their appropriate prompts/skills.

        Args:
            agents_needed (list[str]): The list of agents needed, this list must have user_proxy in it or an error will be generated.

        Returns:
            dict: A dictionary of agent instances.

        """
        agents_map: dict[str, autogen.UserProxyAgent | autogen.AssistantAgent] = {}

        # create user agent
        agents_map["user"] = self._create_user_delegate_agent()
        agents_needed.remove("user")

        # create orchestrator agent
        agents_map["orchestrator"] = self._create_orchestrator_agent()
        agents_needed.remove("orchestrator")

        # create tool_executor agent
        agents_map["tool_executor"] = self._create_tool_executor_agent()
        agents_needed.remove("tool_executor")

        # create tool_execution_manager agent
        agents_map["tool_execution_manager"] = self._create_tool_execution_manager()
        agents_needed.remove("tool_execution_manager")

        # register groupchat
        agents_map["tool_groupchat_manager"] = self._register_groupchat(
            agents_map["tool_execution_manager"],
            agents_map["tool_executor"]
        )

        # create rest optional agents
        for agent_needed in agents_needed:
            if agent_needed == "planner":
                # TODO: Implement the planner agent
                agents_map["planner"] = self._create_planner_agent()
            elif agent_needed == "tool_retriever":
                # TODO Implement the tool retriever agent
                pass
            else:
                raise ValueError(f"Unknown agent type: {agent_needed}")
        return agents_map
    
    def _create_user_delegate_agent(self) -> autogen.UserProxyAgent:
        """
        Create an user delegate agent. This agent is used to initiate the converation, and optionally execute tools

        Returns:
            autogen.UserProxyAgent: An instance of autogen.UserProxyAgent. 
        """
        # def is_planner_termination_message(x: dict[str, str])->bool: # type: ignore
        #      should_terminate = False
        #      function: Any = x.get("function", None)
        #      if function is not None:
        #          return False

        #      content:Any = x.get("content", "")
        #      if content is None:
        #         content = ""
        #         should_terminate = True
        #      else:
        #         try:
        #             content_json = parse_response(content)
        #             _terminate = content_json.get('terminate', "no")
        #             final_response = content_json.get('final_response', None)
        #             if(_terminate == "yes"):
        #                 should_terminate = True
        #                 if final_response:
        #                     notify_planner_messages(final_response, message_type=MessageType.ANSWER)
        #         except json.JSONDecodeError:
        #             logger.error("Error decoding JSON response:\n{content}.\nTerminating..")
        #             should_terminate = True

        #      return should_terminate # type: ignore

        user_delegate_agent = autogen.UserProxyAgent(
            name="user",
            llm_config=None,
            system_message=LLM_PROMPTS["USER_AGENT_PROMPT"],
            is_termination_msg=terminate,
            max_consecutive_auto_reply=self.max_chat_round,
        )
        return user_delegate_agent
    
    def _create_orchestrator_agent(self) -> autogen.AssistantAgent:
        """
        Create an orchestrator agent. This agent is used to orchestrate the conversation between different agents.
        
        Returns:
            autogen.AssistantAgent: An instance of autogen.AssistantAgent.
        """
        orchestrator = autogen.AssistantAgent(
            name="orchestrator",
            system_message=LLM_PROMPTS["ORCHESTRATOR_PROMPT"],
            llm_config=self.llm_config,
            is_termination_msg=terminate,
        )
        return orchestrator
    
    def _create_tool_executor_agent(self):
        """
        Create a UserProxyAgent instance for executing browser control.

        Returns:
            autogen.UserProxyAgent: An instance of UserProxyAgent.

        """
        def is_tool_executor_termination_message(x: dict[str, str])->bool: # type: ignore
             tools_call = x.get("tool_calls", "")
             if tools_call :
                return False
             else:
                return True

        tool_executor_agent = autogen.UserProxyAgent(
            name="tool_executor",
            is_termination_msg=is_tool_executor_termination_message,
            human_input_mode="NEVER",
            llm_config=None,
            max_consecutive_auto_reply=self.max_chat_round,
            code_execution_config={
                "last_n_messages": 1,
                "work_dir": "tools",
                "use_docker": False,
            },
            default_auto_reply="",
        )
        return tool_executor_agent
    
    def _create_tool_execution_manager(self):
        """
        Create a tool execution manager agent.

        Args:
            tool_executor (autogen.UserProxyAgent): The tool executor agent.

        Returns:
            autogen.UserProxyAgent: An instance of UserProxyAgent.

        """
        tool_execution_manager = autogen.AssistantAgent(
            name="tool_execution_manager",
            system_message=LLM_PROMPTS["TOOL_EXECUTION_MANAGER_PROMPT"],
            llm_config=self.llm_config,
        )
        return tool_execution_manager

    def _create_planner_agent(self):
        """
        Create a Planner Agent instance. This is mainly used for exploration at this point

        Returns:
            autogen.AssistantAgent: An instance of PlannerAgent.

        """
        planner_agent = autogen.AssistantAgent(
            name="planner_agent",
            system_message=LLM_PROMPTS["PLANNER_PROMPT"],
            llm_config=self.llm_config,
        )
        # # Register get_user_input skill for LLM by assistant agent
        # planner_agent.register_for_llm(description=LLM_PROMPTS["GET_USER_INPUT_PROMPT"])(get_user_input)
        # # Register get_user_input skill for execution by user_proxy_agent
        # user_proxy_agent.register_for_execution()(get_user_input)

        # self.agent.register_reply( # type: ignore
        #     [autogen.AssistantAgent, None],
        #     reply_func=print_message_as_planner,
        #     config={"callback": None},
        #     ignore_async_in_sync_chat=True
        # )
        return planner_agent

    def _register_groupchat(self, tool_execution_manager, tool_executor):
        """
        Register a group chat for the agents.
        """

        groupchat = autogen.GroupChat(
            agents=[tool_execution_manager, tool_executor],
            messages=[],
            speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
            allow_repeat_speaker=False,
            max_round=8,
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            is_termination_msg=terminate,
            llm_config=None,
            code_execution_config={
                "work_dir": "tool",
                "use_docker": False,
            },
        )
        return manager

    def register_tools(self, apis, use_dummy=False):
        """
        Register all the tools that the agent can perform.
        """
        if use_dummy:
            # register dummy tools for debugging
            def get_list_of_provinces_in_thailand():
                return ["Bangkok", "Chiang Mai", "Phuket", "Krabi", "Pattaya"]
            
            def get_list_of_provinces_in_canada():
                return ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"]
            
            def get_list_of_districts_in_bangkok_thailand():
                return ["Bang Kapi", "Bang Khae", "Bang Khen", "Bang Kho Laem", "Bang Khun Thian"]
            
            def get_list_of_districts_in_toronto_canada():
                return ["Old Toronto", "York", "East York", "North York", "Etobicoke"]
            
            dummy_tools = [
                get_list_of_provinces_in_thailand,
                get_list_of_provinces_in_canada,
                get_list_of_districts_in_bangkok_thailand,
                get_list_of_districts_in_toronto_canada
            ]
            dummy_tool_descriptions = [
                "Fetch the list of provinces in Thailand",
                "Fetch the list of provinces in Canada",
                "Fetch the list of districts in Bangkok",
                "Fetch the list of districts in Toronto"
            ]
            for tool, desc in zip(dummy_tools, dummy_tool_descriptions):
                autogen.register_function(
                    tool,
                    caller=self.agents_map["tool_execution_manager"],  # The assistant agent can suggest calls to the calculator.
                    executor=self.agents_map["tool_executor"],  # The user proxy agent can execute the calculator calls.
                    description=desc,  # A description of the tool.
                )
                print(f">>> Registered tool: {tool.__name__}")
        else:
            for api in apis:
                # convert api to function and add it to the locals
                func_name = api["name"]
                func = convert_api_to_function(api)
                locals()[func_name] = func
                
                autogen.register_function(
                    func,
                    caller=self.agents_map["tool_execution_manager"],  # The assistant agent can suggest calls to the calculator.
                    executor=self.agents_map["tool_executor"],  # The user proxy agent can execute the calculator calls.
                    name=func_name,  # By default, the function name is used as the tool name.
                    description=api["description"],  # A description of the tool.
                )
                # update llm tool signature
                self.agents_map["tool_execution_manager"].update_tool_signature(
                    convert_to_openai_tool_schema(api),
                    is_remove=False)
                print(f">>> Registered tool: {func_name}")
            print(">>> All tools available")
            print(self.agents_map["tool_execution_manager"].llm_config["tools"])
            # breakpoint()
    
    def initiate_chat(self, user_query: str = None):
        res = self.agents_map["user"].initiate_chats(
            [
                {
                    "recipient": self.agents_map["orchestrator"],
                    "message": user_query,
                    "max_turns": 1,
                    "summary_method": "last_msg"
                 },
            ]
        )
    
if __name__ == "__main__":
    autogen_wrapper = AutogenWrapper(max_chat_round=50)
    autogen_wrapper.create(["user", "orchestrator", "tool_executor", "tool_execution_manager"])
    print("AutogenWrapper created successfully")
    
    autogen_wrapper.initiate_chat()
    