# ==============================================================================
# Sensory information in spatial descriptions vs. reference sentences
# ==============================================================================
# This script
#   1. reshapes the sensorimotor / imageability ratings into long format,
#   2. models temporal trends (publication year x corpus) for each sensory
#      modality and for imageability, and plots the estimated trends,
#   3. reports trends and corpus differences (incl. confidence intervals), and
#   4. tests whether each modality predicts imageability differently in the
#      two corpora.
#
# Input files (expected in the working directory):
#   spatial_descriptions.csv, reference.csv
# Output files:
#   temporal_trends_complete.png, temporal_trends_imagability.png
# ==============================================================================

library(tidyverse)
library(emmeans)
library(sjPlot)


# Setup -------------------------------------------------------------------

# Rating dimensions (Lancaster Sensorimotor Norms + imageability)
sensory_dims <- c("olfactory_mean", "auditory_mean", "visual_mean", "haptic_mean")
all_dims     <- c(sensory_dims, "imagability_mean")

# Colours and display labels for the four sensory modalities
dimension_colors <- c(
  auditory_mean  = "#440154",
  olfactory_mean = "#3b528b",
  haptic_mean    = "#5ec962",
  visual_mean    = "#21918c"
)

modality_labels <- c(
  olfactory_mean = "Olfactory",
  auditory_mean  = "Auditory",
  visual_mean    = "Visual",
  haptic_mean    = "Haptic"
)

# Display labels for the two corpora
corpus_labels <- c(
  descriptions = "Spatial Descriptions",
  reference    = "Other Sentences"
)


# Load data ---------------------------------------------------------------

descriptions <- read_csv("spatial_descriptions.csv")
reference    <- read_csv("reference.csv")


# Reshape data ------------------------------------------------------------

# Convert one corpus from wide to long format: one row per sentence x dimension
reshape_long <- function(data, corpus_name) {
  data %>%
    pivot_longer(
      cols      = all_of(all_dims),
      names_to  = "dimension",
      values_to = "value"
    ) %>%
    mutate(
      dimension = as.factor(dimension),
      corpus    = factor(corpus_name)
    )
}

df_long_descriptions <- reshape_long(descriptions, "descriptions")
df_long_reference    <- reshape_long(reference,    "reference")

# Both corpora in one long data frame
df_concat <- bind_rows(df_long_descriptions, df_long_reference) %>%
  dplyr::select(sentence_id, dimension, value, corpus, pub_year)


# Plot helpers ------------------------------------------------------------

# Fit value ~ pub_year * corpus and return the estimated marginal means
# (predicted value + confidence interval) along the publication year.
# plot_model() is only used here to obtain the data behind its plot.
get_emm_data <- function(model) {
  p <- plot_model(
    model,
    type      = "emm",
    terms     = c("pub_year", "corpus"),
    title     = "",
    dodge     = 0.5,
    dot.size  = 0
  )

  dat <- as.data.frame(p$data)
  dat$corpus_type <- factor(
    as.character(dat$group),
    levels = c("descriptions", "reference")
  )
  dat
}

# Line type per corpus (solid = descriptions, dashed = reference)
scale_linetype_corpus <- function() {
  scale_linetype_manual(
    values = c(descriptions = "solid", reference = "dashed"),
    labels = corpus_labels,
    name   = "Dataset"
  )
}

# Shared plot theme
theme_trends <- function() {
  theme_minimal(base_size = 16) +
    theme(
      legend.position      = "right",
      legend.direction     = "vertical",
      legend.text          = element_text(size = 12),
      legend.title         = element_text(size = 14, face = "bold"),
      legend.key.height    = unit(0.5, "cm"),
      legend.key.width     = unit(1, "cm"),
      legend.justification = c(0, 0.5),
      axis.text            = element_text(size = 12),
      axis.title           = element_text(size = 14, face = "bold"),
      panel.grid.minor     = element_blank(),
      plot.margin          = margin(5, 5, 5, 5)
    )
}

# Save a plot with the settings used throughout this script
save_plot <- function(filename, plot) {
  ggsave(
    filename,
    plot   = plot,
    width  = 8,
    height = 4,
    units  = "in",
    dpi    = 300
  )
}


# Temporal trends: sensory modalities -------------------------------------

# For each modality, fit value ~ pub_year * corpus and collect the model-based
# trend lines (with confidence intervals).
emm_by_dimension <- list()

for (dim_name in sensory_dims) {

  model <- glm(
    value ~ pub_year * corpus,
    data   = filter(df_concat, dimension == dim_name),
    family = gaussian
  )

  dat <- get_emm_data(model)
  dat$dimension <- dim_name
  emm_by_dimension[[dim_name]] <- dat
}

dat_all <- bind_rows(emm_by_dimension)

# Plot: colour = modality, line type = corpus, ribbon = 95% CI
p_modalities <- ggplot(
  dat_all,
  aes(
    x        = x,
    y        = predicted,
    color    = dimension,
    linetype = corpus_type,
    group    = interaction(dimension, corpus_type)
  )
) +
  geom_ribbon(
    aes(ymin = conf.low, ymax = conf.high, fill = dimension),
    alpha = 0.15,
    color = NA
  ) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(
    values = dimension_colors,
    labels = modality_labels,
    name   = "Modality"
  ) +
  scale_fill_manual(values = dimension_colors, guide = "none") +
  scale_linetype_corpus() +
  labs(x = "Publication year", y = "Estimated value") +
  theme_trends()

save_plot("temporal_trends_complete.png", p_modalities)
print(p_modalities)


# Temporal trends: imageability -------------------------------------------

model_imag <- glm(
  value ~ pub_year * corpus,
  data   = filter(df_concat, dimension == "imagability_mean"),
  family = gaussian
)

dat_imag <- get_emm_data(model_imag)

# Plot: greyscale, line type = corpus, ribbon = 95% CI
p_imag <- ggplot(
  dat_imag,
  aes(
    x        = x,
    y        = predicted,
    linetype = corpus_type,
    group    = corpus_type
  )
) +
  geom_ribbon(
    aes(ymin = conf.low, ymax = conf.high),
    fill  = "grey70",
    alpha = 0.4,
    color = NA
  ) +
  geom_line(color = "grey20", linewidth = 0.9) +
  scale_linetype_corpus() +
  labs(x = "Publication year", y = "Estimated imagability") +
  theme_trends()

save_plot("temporal_trends_imagability.png", p_imag)
print(p_imag)


# Model output: trends and corpus differences -----------------------------

# For every dimension, fit value ~ pub_year_c * corpus (publication year
# centred at its mean) and report:
#   - the temporal trend within each corpus,
#   - the difference between the two trends (year x corpus interaction),
#   - the corpus difference at the mean, first and last publication year.

# Publication year summary (over both corpora)
mean_year <- mean(df_concat$pub_year, na.rm = TRUE)
min_year  <- min(df_concat$pub_year,  na.rm = TRUE)
max_year  <- max(df_concat$pub_year,  na.rm = TRUE)

# Corpus difference (descriptions - reference) at a given centred year
corpus_difference_at <- function(model, year_c) {
  emmeans(model, ~ corpus, at = list(pub_year_c = year_c)) %>%
    pairs() %>%
    summary(infer = c(TRUE, TRUE))
}

# Build a formatted row (effect, CI, p-value) for the results table
table_row <- function(predictor, estimate, lower, upper, p) {
  tibble(
    Predictor = predictor,
    Effect    = round(estimate, 3),
    CI        = paste0("[", round(lower, 3), ", ", round(upper, 3), "]"),
    p         = format.pval(p, digits = 3, eps = .001)
  )
}

for (dim_name in all_dims) {

  cat("\n=====================================\n")
  cat("Dimension:", dim_name, "\n")
  cat("=====================================\n")

  dat <- df_concat %>%
    filter(dimension == dim_name) %>%
    filter(complete.cases(value, pub_year, corpus)) %>%
    mutate(pub_year_c = pub_year - mean_year)

  model <- glm(
    value ~ pub_year_c * corpus,
    data   = dat,
    family = gaussian
  )

  print(coef(summary(model)))

  # Temporal trend within each corpus
  trends        <- emtrends(model, ~ corpus, var = "pub_year_c")
  trend_results <- summary(trends, infer = c(TRUE, TRUE))

  # Difference between the two trends
  trend_difference <- summary(pairs(trends), infer = c(TRUE, TRUE))

  # Corpus differences at the mean, first and last publication year
  diff_mean  <- corpus_difference_at(model, 0)
  diff_first <- corpus_difference_at(model, min_year - mean_year)
  diff_last  <- corpus_difference_at(model, max_year - mean_year)

  results_table <- bind_rows(
    table_row(
      paste("Publication year - within", trend_results$corpus),
      trend_results$pub_year_c.trend,
      trend_results$lower.CL, trend_results$upper.CL, trend_results$p.value
    ),
    table_row(
      "Publication year \u00d7 Corpus",
      trend_difference$estimate,
      trend_difference$lower.CL, trend_difference$upper.CL, trend_difference$p.value
    ),
    table_row(
      paste0("Corpus difference at mean publication year (", round(mean_year, 1), ")"),
      diff_mean$estimate,
      diff_mean$lower.CL, diff_mean$upper.CL, diff_mean$p.value
    ),
    table_row(
      paste0("Corpus difference at first publication year (", min_year, ")"),
      diff_first$estimate,
      diff_first$lower.CL, diff_first$upper.CL, diff_first$p.value
    ),
    table_row(
      paste0("Corpus difference at last publication year (", max_year, ")"),
      diff_last$estimate,
      diff_last$lower.CL, diff_last$upper.CL, diff_last$p.value
    )
  )

  print(results_table)

  cat("\nFirst publication year:", min_year, "\n")
  cat("Mean publication year:",    round(mean_year, 2), "\n")
  cat("Last publication year:",    max_year, "\n\n")
}


# Sensory information predicting imageability -----------------------------

# Wide-format data (one row per sentence, both corpora stacked).
df <- bind_rows(
  mutate(descriptions, corpus = "descriptions"),
  mutate(reference,    corpus = "reference")
) %>%
  mutate(corpus = factor(corpus))

# For one modality, fit imagability ~ modality * corpus and return
#   - the slope of the modality within each corpus,
#   - the contrast between the two slopes (= interaction).
run_model <- function(modality, data) {
  fit <- glm(
    as.formula(paste("imagability_mean ~", modality, "* corpus")),
    data   = data,
    family = gaussian
  )

  slopes   <- emtrends(fit, "corpus", var = modality)
  contrast <- pairs(slopes, reverse = FALSE)

  list(
    model    = fit,
    slopes   = summary(slopes,   infer = c(TRUE, TRUE)),
    contrast = summary(contrast, infer = c(TRUE, TRUE))
  )
}

results <- set_names(
  map(sensory_dims, run_model, data = df),
  sensory_dims
)

for (modality in sensory_dims) {

  cat("\n=====================================\n")
  cat("Dimension:", modality, "\n")
  cat("=====================================\n\n")

  cat("--- Slopes per dataset ---\n")
  print(results[[modality]]$slopes)

  cat("\n--- Contrast (interaction) ---\n")
  print(results[[modality]]$contrast)
}
