

def get_environment(env_name):
    try:
        if env_name in ('tictactoe'):
            from environments.tictactoe_env import TicTacToeEnv
            return TicTacToeEnv
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py')
    

def get_network_arch(env_name):
    if env_name in ('tictactoe'):
        from models.tictactoe_model import TicTacToePolicy
        return TicTacToePolicy
    else:
        raise Exception(f'No model architectures found for {env_name}')

