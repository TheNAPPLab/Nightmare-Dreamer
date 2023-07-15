        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier, ],
                                          lr=lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost):
        """Penalty loss for Lagrange multiplier."""
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, ep_costs):
        """ Update Lagrange multiplier (lambda)
            Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
            are already averaged across MPI processes.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(ep_costs)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]