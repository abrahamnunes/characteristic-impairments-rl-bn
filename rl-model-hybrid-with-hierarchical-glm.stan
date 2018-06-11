data {
  int<lower=0> N;                 # Number of subjects
  int<lower=0> T;                 # Number of trials
  int<lower=1, upper=2> A1[T, N]; # First step choice
  int<lower=2, upper=3> S2[T, N]; # Second step state
  int<lower=1, upper=2> A2[T, N]; # Second step choice
  int<lower=0, upper=1> BED[N];   # Covariate
  int<lower=0, upper=1> BN[N];    # Covariate
  real R[T, N];                   # Reward/omission vector at each trial
}
transformed data {
    vector[2] Q_o[3];
    vector[2] p_rep_o;

    p_rep_o = rep_vector(0.0, 2);

    for (i in 1:3){
        Q_o[i] = rep_vector(0.0, 2);
    }

}
parameters {
  # GROUP-LEVEL MEANS AND VARIANCES
  vector[6] mu;         # Mean
  vector<lower=0>[6] s; # Variance

  # GLM WEIGHTS
  #
  real K_MB_BN;  # B_MB for covariate
  real K_MF_BN;  # B_MF for covariate
  real K_P_BN;
  real K_B2_BN;
  real K_MB_BED;  # B_MB for covariate
  real K_MF_BED;  # B_MF for covariate
  real K_P_BED;
  real K_B2_BED;

  # SUBJECT-LEVEL WEIGHTS
  #
  real B_MB_raw[N]; # Model-based weight
  real B_MF_raw[N]; # Model-free weight
  real B_P_raw[N];  # Perseveration weight
  real B_2_raw[N];  # Second-step choice randomness
  real LR1_raw[N];   # Unconstrained (raw) learning rate parameter (step1)
  real LR2_raw[N];  # Unconstrained (raw) learning rate parameter (step 2)
}
transformed parameters {
  real B_MB[N]; # Model-based weight
  real B_MF[N]; # Model-free weight
  real B_P[N];  # Perseveration weight
  real B_2[N];  # Second-step choice randomness
  real LR1[N];   # Constrained learning rate parameter (S1)
  real LR2[N];   # Constrained learning rate parameter (S2)

  for (i in 1:N) {
    B_MB[i] = mu[1] + K_MB_BED*BED[i] + K_MB_BN*BN[i] + B_MB_raw[i]*s[1];
    B_MF[i] = mu[2] + K_MF_BED*BED[i] + K_MF_BN*BN[i] + B_MF_raw[i]*s[2];
    B_P[i] = mu[3] + K_P_BED*BED[i] + K_P_BN*BN[i] + B_P_raw[i]*s[3];
    B_2[i] = mu[4] + K_B2_BED*BED[i] + K_B2_BN*BN[i] + B_2_raw[i]*s[4];
    LR1[i] = Phi_approx(mu[5]+LR1_raw[i]*s[5]);
    LR2[i] = Phi_approx(mu[6]+LR2_raw[i]*s[6]);
  }
}
model {
  # SAMPLE GROUP-LEVEL HYPERPARAMETERS
  mu ~ cauchy(0, 5);
  s ~ cauchy(0, 5);

  # SAMPLE SUBJECT-LEVEL PARAMETERS
  B_MB_raw ~ normal(0, 1);
  B_MF_raw ~ normal(0, 1);
  B_P_raw ~ normal(0, 1);
  B_2_raw ~ normal(0, 1);
  LR1_raw ~ normal(0, 1);
  LR2_raw ~ normal(0, 1);

  # RUN SUBJECT LEVEL MODELS
  for (i in 1:N) {
      vector[2] Qmf[3];
      vector[2] Qmb[3];
      vector[2] p_rep;
      real PE2;
      real PE1;

      Qmf = Q_o;
      Qmb = Q_o;

      p_rep = p_rep_o;

      for (t in 1:T) {
          # First state action
          A1[t, i] ~ categorical_logit(B_MB[i]*Qmb[1] + B_MF[i]*Qmf[1] + B_P[i]*p_rep);

          # Second state action
          A2[t, i] ~ categorical_logit(B_2[i]*Qmf[S2[t, i]]);

          # Prediction error
          PE2 = R[t, i] - Qmf[S2[t, i]][A2[t, i]];
          PE1 = Qmf[S2[t, i]][A2[t, i]] - Qmf[1][A1[t, i]];

          # Decay Q
          Qmf[1] = (1-LR1[i])*Qmf[1];
          Qmf[2] = (1-LR2[i])*Qmf[2];
          Qmf[3] = (1-LR2[i])*Qmf[3];

          # Learning (Model Free)
          Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + LR1[i]*PE1;
          Qmf[S2[t, i]][A2[t, i]] = Qmf[S2[t, i]][A2[t, i]] + LR2[i]*PE2;
          Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + LR1[i]*PE2;

          # Learning (Model Based)
          Qmb[1][1] = 0.7*max(Qmf[2]) + 0.3*max(Qmf[3]);
          Qmb[1][2] = 0.7*max(Qmf[3]) + 0.3*max(Qmf[2]);

          # Compute the REP() function
          for (j in 1:2) {
            if (A1[t, i] == j) {
              p_rep[j] = 1;
            } else {
              p_rep[j] = 0;
            }
          }
      }
  }
}
