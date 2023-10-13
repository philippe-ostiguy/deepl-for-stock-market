import torch

def get_risk_rewards_metrics(daily_returns : torch.Tensor) -> dict:
    metrics = {}
    if daily_returns.shape[0] <= 15:
        metrics['annualized_return'] = torch.tensor(0.0)
        metrics['annualized_risk'] = torch.tensor(0.0)
        metrics['return_on_risk'] = torch.tensor(0.0)
        return metrics

    metrics['annualized_return'] = torch.prod(1 + daily_returns) ** (
            252.0 / daily_returns.shape[0]) - 1
    metrics['annualized_risk'] = daily_returns.std() * (252 ** 0.5)
    if metrics['annualized_risk'].item() < 1e-6:
        metrics['return_on_risk'] = torch.tensor(0.0, dtype=torch.float)
    else:
        metrics['return_on_risk'] = metrics['annualized_return'] / metrics['annualized_risk']

    return metrics