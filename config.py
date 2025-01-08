from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="TRAILCAML",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[
        Validator("DATASET_DIR", default="data/trailcam-dataset"),
    ],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
