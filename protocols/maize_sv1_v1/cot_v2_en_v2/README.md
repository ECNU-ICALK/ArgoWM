# Maize sv1 English CoT protocol v2

Teacher prompt 2.2.1 clarifies that stage labels belong only in stage_reasoning; phenotype and growth-plan text must remain qualitative and contain no digits or stage codes. The provider-input builder now includes strict_output_rules in the actual system prompt.