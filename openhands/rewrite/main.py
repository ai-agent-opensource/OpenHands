
import uuid
import asyncio
import time
from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import (
    ConversationStoreImpl,
    SecretsStoreImpl,
    SettingsStoreImpl,
    config,
    conversation_manager,
    server_config,
)
from openhands.cli.main import run_setup_flow
from openhands.server.session.conversation_init_data import ConversationInitData

from openhands.server.routes.manage_conversations import (
    ConversationResponse,
    InitSessionRequest,
    delete_conversation,
    get_conversation,
    new_conversation,
    search_conversations,
)

# from openhands.server.session.session import ROOM_KEY, Session
import socketio
from openhands.storage.files import FileStore

from openhands.server.session.agent_session import AgentSession
from openhands.core.config import OpenHandsConfig
from logging import LoggerAdapter

from openhands.events.action import MessageAction, NullAction
from openhands.storage.data_models.settings import Settings
from openhands.core.logger import OpenHandsLoggerAdapter
from copy import deepcopy

file_store = FileStore
sio = socketio.AsyncServer


class Session:
    sid: str
    sio: socketio.AsyncServer | None
    last_active_ts: int = 0
    is_alive: bool = True
    agent_session: AgentSession
    loop: asyncio.AbstractEventLoop
    config: OpenHandsConfig
    file_store: FileStore
    user_id: str | None
    logger: LoggerAdapter

    def __init__(
        self,
        sid: str,
        config: OpenHandsConfig,
        file_store: FileStore,
        sio: socketio.AsyncServer | None,
        user_id: str | None = None,
    ):
        self.sid = sid
        self.sio = sio
        self.last_active_ts = int(time.time())
        self.file_store = file_store
        self.logger = OpenHandsLoggerAdapter(extra={'session_id': sid})
        self.agent_session = AgentSession(
            sid,
            file_store,
            status_callback=self.queue_status_message,
            user_id=user_id,
        )
        self.agent_session.event_stream.subscribe(
            EventStreamSubscriber.SERVER, self.on_event, self.sid
        )
        # Copying this means that when we update variables they are not applied to the shared global configuration!
        self.config = deepcopy(config)
        self.loop = asyncio.get_event_loop()
        self.user_id = user_id

    async def initialize_agent(
        self,
        settings: Settings,
        initial_message: MessageAction | None,
        replay_json: str | None,
    ) -> None:
        self.logger.info(f'Initialize agent')
        exit(0)

        try:
            await self.agent_session.start(
                runtime_name=self.config.runtime,
                config=self.config,
                agent=agent,
                max_iterations=max_iterations,
                max_budget_per_task=max_budget_per_task,
                agent_to_llm_config=self.config.get_agent_to_llm_config_map(),
                agent_configs=self.config.get_agent_configs(),
                git_provider_tokens=git_provider_tokens,
                custom_secrets=custom_secrets,
                selected_repository=selected_repository,
                selected_branch=selected_branch,
                initial_message=initial_message,
                conversation_instructions=conversation_instructions,
                replay_json=replay_json,
            )
        except MicroagentValidationError as e:
            self.logger.exception(f'Error creating agent_session: {e}')
            # For microagent validation errors, provide more helpful information
            await self.send_error(f'Failed to create agent session: {str(e)}')
            return
        except ValueError as e:
            self.logger.exception(f'Error creating agent_session: {e}')
            error_message = str(e)
            # For ValueError related to microagents, provide more helpful information
            if 'microagent' in error_message.lower():
                await self.send_error(
                    f'Failed to create agent session: {error_message}'
                )
            else:
                # For other ValueErrors, just show the error class
                await self.send_error('Failed to create agent session: ValueError')
            return
        except Exception as e:
            self.logger.exception(f'Error creating agent_session: {e}')
            # For other errors, just show the error class to avoid exposing sensitive information
            await self.send_error(
                f'Failed to create agent session: {e.__class__.__name__}'
            )
            return

    async def _send_status_message(self, msg_type: str, id: str, message: str) -> None:
        """Sends a status message to the client."""
        if msg_type == 'error':
            agent_session = self.agent_session
            controller = self.agent_session.controller
            if controller is not None and not agent_session.is_closed():
                await controller.set_agent_state_to(AgentState.ERROR)
            self.logger.error(
                f'Agent status error: {message}',
                extra={'signal': 'agent_status_error'},
            )
        await self.send(
            {'status_update': True, 'type': msg_type, 'id': id, 'message': message}
        )

    def queue_status_message(self, msg_type: str, id: str, message: str) -> None:
        """Queues a status message to be sent asynchronously."""
        asyncio.run_coroutine_threadsafe(
            self._send_status_message(msg_type, id, message), self.loop
        )

_local_agent_loops_by_sid: dict[str, Session] = {}


async def get_running_agent_loops(
        user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> set[str]:
        """Get the running session ids in chronological order (oldest first).

        If a user is supplied, then the results are limited to session ids for that user.
        If a set of filter_to_sids is supplied, then results are limited to these ids of interest.

        Returns:
            A set of session IDs
        """
        # Get all items and convert to list for sorting
        items: Iterable[tuple[str, Session]] = _local_agent_loops_by_sid.items()

        # Filter items if needed
        if filter_to_sids is not None:
            items = (item for item in items if item[0] in filter_to_sids)
        if user_id:
            items = (item for item in items if item[1].user_id == user_id)

        sids = {sid for sid, _ in items}
        return sids

async def main():
    user_id = None
    git_provider_tokens = None
    custom_secrets = None
    selected_repository = None
    selected_branch = None
    initial_user_msg = None
    image_urls = None
    replay_json = None
    conversation_trigger = {"value" : None}
    conversation_instructions = None
    git_provider = None
    conversation_id = None

    logger.info('Loading settings')
    settings_store = await SettingsStoreImpl.get_instance(config, user_id)
    settings = await settings_store.load()
    logger.info('Settings loaded')

    # add for easy setup in terminal
    if not settings:
        # Clear the terminal before showing the banner
        await run_setup_flow(config, settings_store)
        banner_shown = True
        settings = await settings_store.load()

    session_init_args: dict[str, Any] = {}
    session_init_args['git_provider_tokens'] = git_provider_tokens
    session_init_args['selected_repository'] = selected_repository
    session_init_args['custom_secrets'] = custom_secrets
    session_init_args['selected_branch'] = selected_branch
    session_init_args['git_provider'] = git_provider
    session_init_args['conversation_instructions'] = conversation_instructions
    conversation_init_data = ConversationInitData(**session_init_args)

    logger.info('Loading conversation store')
    conversation_store = await ConversationStoreImpl.get_instance(config, user_id)
    logger.info('ServerConversation store loaded')

    if conversation_id is None:
        unique_id = uuid.uuid4().hex
        conversation_id = unique_id
        sid = unique_id

    # if not await conversation_store.exists(conversation_id):
    #     logger.info(
    #         f'New conversation ID: {conversation_id}',
    #         extra={'user_id': user_id, 'session_id': conversation_id},
    #     )

    #     conversation_init_data = ExperimentManagerImpl.run_conversation_variant_test(
    #         user_id, conversation_id, conversation_init_data
    #     )
    #     conversation_title = get_default_conversation_title(conversation_id)

    #     logger.info(f'Saving metadata for conversation {conversation_id}')
    #     await conversation_store.save_metadata(
    #         ConversationMetadata(
    #             trigger=conversation_trigger,
    #             conversation_id=conversation_id,
    #             title=conversation_title,
    #             user_id=user_id,
    #             selected_repository=selected_repository,
    #             selected_branch=selected_branch,
    #             git_provider=git_provider,
    #             llm_model=conversation_init_data.llm_model,
    #         )
    #     )


    logger.info(
        f'Starting agent loop for conversation {conversation_id}',
        extra={'user_id': user_id, 'session_id': conversation_id},
    )

    # agent_loop_info = await conversation_manager.maybe_start_agent_loop(
    #     conversation_id,
    #     conversation_init_data,
    #     user_id,
    #     initial_user_msg=initial_message_action,
    #     replay_json=replay_json,
    # )

    logger.info(f'maybe_start_agent_loop:{sid}', extra={'session_id': sid})
    session = _local_agent_loops_by_sid.get(sid)

    # if not session:
    #     session = await self._start_agent_loop(
    #         sid, settings, user_id, initial_user_msg, replay_json
    #     )

    logger.info(f'starting_agent_loop:{sid}', extra={'session_id': sid})
    response_ids = await get_running_agent_loops(user_id)

    status_response = await get_conversation(conversation_id=conversation_id)
    current_status = status_response.conversation_status
    print(f"현재 대화 상태: {current_status}")



    if len(response_ids) >= config.max_concurrent_conversations:
        logger.info(
            f'too_many_sessions_for:{user_id or ""}',
            extra={'session_id': sid, 'user_id': user_id},
        )

    session = Session(
        sid=sid,
        file_store=file_store,
        config=config,
        sio=sio,
        user_id=user_id,
    )

    _local_agent_loops_by_sid[sid] = session
    asyncio.create_task(
        session.initialize_agent(settings, initial_user_msg, replay_json)
    )



if __name__ == "__main__":
    asyncio.run(main())
