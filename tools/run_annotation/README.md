# Annotation

This program presents an annotation interface to annotate experiment logs.

## Get Started

On the root directory of `DeformPAM`, run the following command:

```bash
# annotate the supervised learning data
make scripts.run_supervised_annotation 
# annotate the preference learning data
make scripts.run_finetune_sort_annotation
```

There are several configurable parameters, most of them a intuitive, the `api_url` should be set to the url of Service Backend Module (See ![../data_management/README.md](../data_management/README.md))
