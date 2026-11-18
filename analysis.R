# Data Analysis


library(tidyverse)
library(emmeans)
library(sjPlot)

descriptions <- read_csv("spatial_descriptions.csv")
reference    <- read_csv("reference.csv")

# Reshape data ------------------------------------------------------------

df_long_descriptions <- descriptions %>%
    pivot_longer(cols = c("olfactory_mean", "auditory_mean", "haptic_mean", "visual_mean", "imagability_mean"),
                 names_to = "dimension",
                 values_to = "value")

df_long_descriptions <- df_long_descriptions %>%
    mutate(dimension = as.factor(dimension))


df_long_reference <- reference %>%
    pivot_longer(cols = c("olfactory_mean", "auditory_mean", "haptic_mean", "visual_mean", "imagability_mean"),
                 names_to = "dimension",
                 values_to = "value")

df_long_reference <- df_long_reference %>%
    mutate(dimension = as.factor(dimension))


df_long_descriptions <- df_long_descriptions %>%
    mutate(corpus = "descriptions")%>%
    mutate(corpus = as.factor(corpus))

df_long_reference <- df_long_reference %>%
    mutate(corpus = "reference")%>%
    mutate(corpus = as.factor(corpus))


df_concat <- bind_rows(df_long_descriptions, df_long_reference) %>%
     dplyr::select(sentence_id, dimension, value, corpus, pub_year)

# Strength of Sensory Modalities and Imageability -------------------------

# Lancaster Sensorimotor Norms

dimension_colors <- c(
  "auditory_mean" = "#440154",
  "olfactory_mean"  = "#3b528b",
  "haptic_mean"    = "#5ec962",
  "visual_mean"    = "#21918c"
)

dimensions <- names(dimension_colors)

all_data <- list()


for (dimension in dimensions) {

  model <- glm(
    value ~ pub_year * corpus,
    data = df_concat %>%
      filter(dimension == !!dimension),
    family = gaussian
  )

  # print(paste("Summary for dimension:", dimension))
  # print(summary(model))

  p <- plot_model(
    model,
    type = "emm",
    terms = c("pub_year", "corpus"),
    title = "",
    colors = c("#1B9E77", "#D95F02"),
    dodge = 0.5,
    dot.size = 0
  )

  dat <- as.data.frame(p$data)
  dat$dimension <- dimension
  all_data[[dimension]] <- dat
}


# combine all modalities
dat_all <- do.call(rbind, all_data)
dat_all <- as.data.frame(dat_all)
dat_all$corpus_type <- as.character(dat_all$group)
dat_all$corpus_type <- factor(
  dat_all$corpus_type,
  levels = c("descriptions", "reference")
)


p <- ggplot(
  dat_all,
  aes(
    x = x,
    y = predicted,

    # dimension
    color = dimension,

    # corpus
    linetype = corpus_type,

    group = interaction(dimension, corpus_type)
  )
) +



# confidence intervals
  geom_ribbon(
    aes(
      ymin = conf.low,
      ymax = conf.high,
      fill = dimension
    ),
    alpha = 0.15,
    color = NA
  ) +

# lines
  geom_line(
    linewidth = 0.9
  ) +

# dimension colors
  scale_color_manual(
    values = dimension_colors,
    name = "Modality",
    labels = c(
      "olfactory_mean" = "Olfactory",
      "auditory_mean"  = "Auditory",
      "visual_mean"    = "Visual",
      "haptic_mean"    = "Haptic"
    )
  ) +

  scale_fill_manual(
    values = dimension_colors,
    guide = "none"
  ) +


# corpus linetypes
  scale_linetype_manual(
    values = c(
      "descriptions" = "solid",
      "reference" = "dashed"
    ),
    labels = c(
      "descriptions" = "Spatial Descriptions",
      "reference" = "Other Sentences"
    ),
    name = "Dataset"
  ) +


  labs(
    x = "Publication year",
    y = "Estimated value"
  ) +

  theme_minimal(base_size = 16) +

  theme(
    legend.position = "right",
    legend.direction = "vertical",

    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14, face = "bold"),

    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold"),

    panel.grid.minor = element_blank(),

    plot.margin = margin(5, 5, 5, 5),

    legend.key.height = unit(0.5, "cm"),
    legend.key.width = unit(1, "cm"),

    legend.justification = c(0, 0.5)
  )


ggsave(
  "temporal_trends_complete.png",
  plot = p,
  width = 8,
  height = 4,
  units = "in",
  dpi = 300
)

print(p)

# Imageability ------------------------------------------------------------

model <- glm(
  value ~ pub_year * corpus,
  data = df_concat %>%
    filter(dimension == "imagability_mean"),
  family = gaussian
)

p_emm <- plot_model(
  model,
  type = "emm",
  terms = c("pub_year", "corpus"),
  title = "",
  colors = c("grey30", "grey30"),
  dodge = 0.5,
  dot.size = 0
)

dat <- as.data.frame(p_emm$data)


dat$corpus_type <- as.character(dat$group)

dat$corpus_type <- factor(
  dat$corpus_type,
  levels = c("descriptions", "reference")
)

p <- ggplot(
  dat,
  aes(
    x = x,
    y = predicted,

    linetype = corpus_type,

    group = corpus_type
  )
) +


  geom_ribbon(
    aes(
      ymin = conf.low,
      ymax = conf.high
    ),
    fill = "grey70",
    alpha = 0.4,
    color = NA
  ) +


  geom_line(
    color = "grey20",
    linewidth = 0.9
  ) +

  scale_linetype_manual(
    values = c(
      "descriptions" = "solid",
      "reference" = "dashed"
    ),
    labels = c(
      "descriptions" = "Spatial Descriptions",
      "reference" = "Other Sentences"
    ),
    name = "Dataset"
  ) +

  labs(
    x = "Publication year",
    y = "Estimated imagability"
  ) +

  theme_minimal(base_size = 16) +

  theme(
    legend.position = "right",
    legend.direction = "vertical",

    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14, face = "bold"),

    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold"),

    panel.grid.minor = element_blank(),

    plot.margin = margin(5, 5, 5, 5),

    legend.key.height = unit(0.5, "cm"),
    legend.key.width = unit(1, "cm"),

    legend.justification = c(0, 0.5)
  )

ggsave(
  "temporal_trends_imagability.png",
  plot = p,
  width = 8,
  height = 4,
  units = "in",
  dpi = 300
)

print(p)

# Model output ------------------------------------------------------------

dimensions <- c(
  "olfactory_mean",
  "auditory_mean",
  "visual_mean",
  "haptic_mean",
  "imagability_mean"
)

mean_year <- mean(
  df_concat$pub_year,
  na.rm = TRUE
)

min_year <- min(
  df_concat$pub_year,
  na.rm = TRUE
)

max_year <- max(
  df_concat$pub_year,
  na.rm = TRUE
)


for (dim_name in dimensions) {

  cat("\n=====================================\n")
  cat("Dimension:", dim_name, "\n")
  cat("=====================================\n")


  dat <- df_concat %>%
    dplyr::filter(dimension == dim_name) %>%
    dplyr::filter(
      complete.cases(value, pub_year, corpus)
    ) %>%
    dplyr::mutate(
      pub_year_c = pub_year - mean_year
    )

  model <- glm(
    value ~ pub_year_c * corpus,
    data = dat,
    family = gaussian
  )


  print(coef(summary(model)))


  # temporal trends within each corpus
  trends <- emtrends(
    model,
    ~ corpus,
    var = "pub_year_c"
  )

  trend_results <- summary(
    trends,
    infer = c(TRUE, TRUE)
  )

  # difference in temporal trends between corpora
  trend_difference <- summary(
    pairs(trends),
    infer = c(TRUE, TRUE)
  )


# corpus differences in mean publication year
  corpus_means <- emmeans(
    model,
    ~ corpus,
    at = list(pub_year_c = 0)
  )

  corpus_difference_mean <- summary(
    pairs(corpus_means),
    infer = c(TRUE, TRUE)
  )


# corpus differences in first publication year
  corpus_first_year <- emmeans(
    model,
    ~ corpus,
    at = list(
      pub_year_c = min_year - mean_year
    )
  )

  corpus_difference_first <- summary(
    pairs(corpus_first_year),
    infer = c(TRUE, TRUE)
  )


# corpus differences in last publication year
  corpus_last_year <- emmeans(
    model,
    ~ corpus,
    at = list(
      pub_year_c = max_year - mean_year
    )
  )

  corpus_difference_last <- summary(
    pairs(corpus_last_year),
    infer = c(TRUE, TRUE)
  )


  results_table <- bind_rows(

    # temporal trends within each corpus
    trend_results %>%
      mutate(
        effect = "Publication year",
        contrast = paste("within", corpus)
      ) %>%
      transmute(
        Predictor = paste(effect, "-", contrast),
        Effect = round(pub_year_c.trend, 3),
        CI = paste0(
          "[", round(lower.CL, 3),
          ", ", round(upper.CL, 3), "]"
        ),
        p = format.pval(
          p.value,
          digits = 3,
          eps = .001
        )
      ),

    # difference in temporal trends between corpora
    trend_difference %>%
      transmute(
        Predictor = "Publication year × Corpus",
        Effect = round(estimate, 3),
        CI = paste0(
          "[", round(lower.CL, 3),
          ", ", round(upper.CL, 3), "]"
        ),
        p = format.pval(
          p.value,
          digits = 3,
          eps = .001
        )
      ),

    # corpus differences in mean publication year
    corpus_difference_mean %>%
      transmute(
        Predictor = paste0(
          "Corpus difference at mean publication year (",
          round(mean_year, 1),
          ")"
        ),
        Effect = round(estimate, 3),
        CI = paste0(
          "[", round(lower.CL, 3),
          ", ", round(upper.CL, 3), "]"
        ),
        p = format.pval(
          p.value,
          digits = 3,
          eps = .001
        )
      ),

    # corpus differences in first publication year
    corpus_difference_first %>%
      transmute(
        Predictor = paste0(
          "Corpus difference at first publication year (",
          min_year,
          ")"
        ),
        Effect = round(estimate, 3),
        CI = paste0(
          "[", round(lower.CL, 3),
          ", ", round(upper.CL, 3), "]"
        ),
        p = format.pval(
          p.value,
          digits = 3,
          eps = .001
        )
      ),

    # corpus differences in last publication year
    corpus_difference_last %>%
      transmute(
        Predictor = paste0(
          "Corpus difference at last publication year (",
          max_year,
          ")"
        ),
        Effect = round(estimate, 3),
        CI = paste0(
          "[", round(lower.CL, 3),
          ", ", round(upper.CL, 3), "]"
        ),
        p = format.pval(
          p.value,
          digits = 3,
          eps = .001
        )
      )
  )

  print(results_table)

  cat("\nFirst publication year:", min_year, "\n")
  cat("Mean publication year:", round(mean_year, 2), "\n")
  cat("Last publication year:", max_year, "\n")

  cat("\n")
}

# Sensory Information predicting Imageability -----------------------------

modalities <- c("olfactory_mean", "auditory_mean", "visual_mean", "haptic_mean")

df$corpus <- factor(df$corpus)

run_model <- function(mod) {
  f   <- as.formula(paste("imagability_mean ~", mod, "* corpus"))
  fit <- glm(f, data = df)

  # slope of modality per corpus
  tr <- emtrends(fit, "corpus", var = mod)
  slopes <- summary(tr, infer = c(TRUE, TRUE))

  # interaction / difference of slopes
  contr <- pairs(tr, reverse = FALSE)
  contr_sum <- summary(contr, infer = c(TRUE, TRUE))

  list(model = fit, slopes = slopes, contrast = contr_sum)
}

results <- set_names(map(modalities, run_model), modalities)

for (mod in modalities) {
  cat("\n=====================================\n")
  cat("Dimension:", mod, "\n")
  cat("=====================================\n\n")

  cat("--- Slopes pro Dataset ---\n")
  print(results[[mod]]$slopes)

  cat("\n--- Contrast (Interaction) ---\n")
  print(results[[mod]]$contrast)
}
