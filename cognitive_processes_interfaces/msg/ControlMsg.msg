# Possible values for command:
# - reset_world: notifies that the world has been reset. The main loop resets it through
#   the world reset service (Control: world_reset_service) before publishing this command.
# - publish_ltm
# - pause
# - continue
string command
string world
int32 iteration