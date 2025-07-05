import asyncio
from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import (
    ConversationStoreImpl,
    SecretsStoreImpl,
    SettingsStoreImpl,
    config,
    conversation_manager,
    server_config,
)

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



if __name__ == "__main__":
    asyncio.run(main())
