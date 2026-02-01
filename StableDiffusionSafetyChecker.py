import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel(config)
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        self.register_buffer("concept_embeds", torch.ones(1, 17, config.projection_dim))
        self.register_buffer("special_care_embeds", torch.ones(1, 3, config.projection_dim))
        self.register_buffer("concept_embeds_weights", torch.ones(1, 17))
        self.register_buffer("special_care_embeds_weights", torch.ones(1, 3))

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(pooled_output)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        special_cos_dist = torch.mm(image_embeds, self.special_care_embeds[0].t())
        cos_dist = torch.mm(image_embeds, self.concept_embeds[0].t())

        has_nsfw_concepts = []
        for i in range(image_embeds.shape[0]):
            concept_idx = (cos_dist[i] > self.concept_embeds_weights[0]).any().item()
            has_nsfw_concepts.append(concept_idx)
            if concept_idx:
                images[i] = torch.zeros_like(images[i])
        
        return images, has_nsfw_concepts
