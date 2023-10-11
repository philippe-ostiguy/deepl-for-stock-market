import torch

def get_risk_rewards_metrics(daily_returns : torch.Tensor) -> dict:
    metrics = {}
    if daily_returns.shape[0] == 0:
        metrics['annualized_return'] = 0
        metrics['annualized_risk'] = 0
        metrics['return_on_risk'] = torch.tensor(0.0)
        return metrics

    metrics['annualized_return'] = annualized_return = torch.prod(1 + daily_returns) ** (
            252.0 / daily_returns.shape[0]) - 1
    metrics['annualized_risk'] = annualized_risk = daily_returns.std() * (252 ** 0.5)
    metrics['return_on_risk'] = annualized_return / annualized_risk if annualized_risk != 0 else torch.tensor(
        0.0)
    return metrics