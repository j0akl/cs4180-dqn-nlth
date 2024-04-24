import csv
from rlcard.utils import print_card
import matplotlib.pyplot as plt


def tournament(env, num, output_filename=None):
    """Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    """
    payoffs_sum_model = []
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        print("GAME NUMBER: ", counter + 1)
        trajectories, _payoffs = env.run(is_training=False)
        # human interface
        final_state = trajectories[0][-1]
        action_record = final_state["action_record"]
        state = final_state["raw_obs"]
        _action_list = []
        for i in range(1, len(action_record) + 1):
            if action_record[-i][0] == state["current_player"]:
                break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print(">> Player", pair[0], "chooses", pair[1])

        # Let's take a look at what the agent card is
        print("===============     Cards all Players    ===============")
        for hands in env.get_perfect_information()["hand_cards"]:
            print_card(hands)

        print("===============     Board Cards    ===============")

        print_card(state["public_cards"])

        print("===============     Result     ===============")
        if _payoffs[0] > 0:
            print("You win {} chips!".format(_payoffs[0]))
        elif _payoffs[0] == 0:
            print("It is a tie.")
        else:
            print("You lose {} chips!".format(-_payoffs[0]))
        print("")

        payoffs_sum_model.append(_payoffs[1])

        input("Press any key to continue...")
        # end human interface
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter

    if output_filename is not None and output_filename != "":
        with open(output_filename, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["hand", "ai_winnings"])
            for i in range(len(payoffs_sum_model)):
                writer.writerow([i, payoffs_sum_model[i]])

    return payoffs
