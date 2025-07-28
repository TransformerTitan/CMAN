# Collaborative Multimodal Agent Networks (CMAN)
# Implementation based on "Collaborative Multimodal Agent Networks: Dynamic Specialization and Emergent Communication for Complex Scene Understanding"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models
import torchaudio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
@dataclass
class CMANConfig:
    # Model architecture
    num_agents: int = 4
    hidden_dim: int = 768
    specialization_dim: int = 64
    message_dim: int = 256
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 30000
    
    # Training parameters
    learning_rate: float = 2e-4
    batch_size: int = 32
    pretrain_epochs: int = 50
    specialization_epochs: int = 100
    warmup_steps: int = 1000
    
    # Loss weights
    lambda_comm: float = 0.1
    lambda_div: float = 0.05
    lambda_reg: float = 0.01
    lambda_max: float = 1.0
    tau_threshold: float = 2.0
    gamma: float = 0.1
    
    # Communication parameters
    beta_kl: float = 0.01
    dropout: float = 0.1
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MultimodalEncoder(nn.Module):
    """Multimodal encoder for processing vision, audio, and text inputs"""
    
    def __init__(self, config: CMANConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder (ResNet-50 based)
        self.vision_backbone = models.resnet50(pretrained=True)
        self.vision_backbone.fc = nn.Identity()
        self.vision_proj = nn.Linear(2048, config.hidden_dim)
        
        # Audio encoder (1D CNN)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, config.hidden_dim)
        )
        
        # Text encoder (transformer-based)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout
            ),
            num_layers=6
        )
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Cross-modal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, vision_input, audio_input, text_input):
        # Process each modality
        if vision_input is not None:
            vision_features = self.vision_backbone(vision_input)
            vision_features = self.vision_proj(vision_features)
        else:
            vision_features = torch.zeros(vision_input.size(0), self.config.hidden_dim).to(self.config.device)
            
        if audio_input is not None:
            audio_features = self.audio_encoder(audio_input)
        else:
            audio_features = torch.zeros(audio_input.size(0), self.config.hidden_dim).to(self.config.device)
            
        if text_input is not None:
            text_embeddings = self.text_embedding(text_input)
            text_features = self.text_encoder(text_embeddings.transpose(0, 1)).mean(dim=0)
        else:
            text_features = torch.zeros(text_input.size(0), self.config.hidden_dim).to(self.config.device)
        
        # Concatenate and fuse modalities
        multimodal_input = torch.stack([vision_features, audio_features, text_features], dim=1)
        fused_output, _ = self.fusion_layer(
            multimodal_input, multimodal_input, multimodal_input
        )
        
        # Final representation
        output = self.layer_norm(fused_output.mean(dim=1))
        return self.dropout(output)

class CommunicationModule(nn.Module):
    """Module for inter-agent communication with emergent protocols"""
    
    def __init__(self, config: CMANConfig):
        super().__init__()
        self.config = config
        
        # Message generation
        self.message_generator = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Message embedding space
        self.message_proj = nn.Linear(config.hidden_dim, config.message_dim)
        self.message_embedding = nn.Linear(config.message_dim, config.hidden_dim)
        
        # Routing mechanism
        self.routing_weights = nn.Parameter(torch.randn(config.specialization_dim, config.specialization_dim))
        
        self.layer_norm = nn.LayerNorm(config.message_dim)
        
    def generate_message(self, hidden_state, peer_states):
        """Generate message using attention over internal representations"""
        # Self-attention to generate message
        message, _ = self.message_generator(
            hidden_state.unsqueeze(0),
            peer_states,
            peer_states
        )
        
        # Project to message space
        message = self.message_proj(message.squeeze(0))
        return self.layer_norm(message)
    
    def compute_routing_scores(self, sender_spec, receiver_specs):
        """Compute attention weights for message routing"""
        scores = torch.matmul(sender_spec.unsqueeze(0), self.routing_weights)
        scores = torch.matmul(scores, receiver_specs.T)
        return F.softmax(scores, dim=-1)
    
    def forward(self, hidden_states, specializations):
        """Process communication between agents"""
        batch_size, num_agents, hidden_dim = hidden_states.shape
        messages = []
        
        for i in range(num_agents):
            # Generate message for agent i
            sender_state = hidden_states[:, i]
            peer_states = torch.cat([
                hidden_states[:, :i], 
                hidden_states[:, i+1:]
            ], dim=1).transpose(0, 1)
            
            message = self.generate_message(sender_state, peer_states)
            messages.append(message)
        
        messages = torch.stack(messages, dim=1)
        
        # Compute routing and aggregate messages
        aggregated_messages = []
        for i in range(num_agents):
            routing_scores = self.compute_routing_scores(
                specializations[:, i], 
                specializations[:, torch.arange(num_agents) != i]
            )
            
            # Aggregate messages from other agents
            other_messages = torch.cat([
                messages[:, :i], 
                messages[:, i+1:]
            ], dim=1)
            
            aggregated = torch.sum(
                routing_scores.unsqueeze(-1) * other_messages, 
                dim=1
            )
            aggregated_messages.append(aggregated)
        
        aggregated_messages = torch.stack(aggregated_messages, dim=1)
        
        # Embed back to hidden space
        embedded_messages = self.message_embedding(aggregated_messages)
        
        return embedded_messages, messages

class Agent(nn.Module):
    """Individual agent with specialization capabilities"""
    
    def __init__(self, config: CMANConfig, agent_id: int):
        super().__init__()
        self.config = config
        self.agent_id = agent_id
        
        # Core components
        self.encoder = MultimodalEncoder(config)
        self.communication = CommunicationModule(config)
        
        # Specialization vector
        self.specialization = nn.Parameter(
            torch.randn(config.specialization_dim) * 0.1
        )
        
        # Decision module
        self.decision_layers = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # *2 for hidden + message
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)  # Task-specific output
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vision_input, audio_input, text_input, peer_messages=None):
        # Encode multimodal input
        hidden_state = self.encoder(vision_input, audio_input, text_input)
        
        # Incorporate peer messages if available
        if peer_messages is not None:
            combined_input = torch.cat([hidden_state, peer_messages], dim=-1)
        else:
            combined_input = torch.cat([hidden_state, torch.zeros_like(hidden_state)], dim=-1)
        
        # Generate decision
        decision = self.decision_layers(combined_input)
        
        # Estimate confidence
        confidence = self.confidence_estimator(hidden_state)
        
        return {
            'hidden_state': hidden_state,
            'decision': decision,
            'confidence': confidence,
            'specialization': self.specialization
        }

class DynamicAgentCoordination(nn.Module):
    """Dynamic Agent Coordination (DAC) algorithm implementation"""
    
    def __init__(self, config: CMANConfig):
        super().__init__()
        self.config = config
        
        # Role assignment network
        self.role_assignment = nn.Sequential(
            nn.Linear(config.specialization_dim + config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_agents),
            nn.Softmax(dim=-1)
        )
        
        # Task embedding
        self.task_embedding = nn.Parameter(torch.randn(config.specialization_dim))
        
        # Arbitration network for conflict resolution
        self.arbitration_network = nn.Sequential(
            nn.Linear(config.num_agents * 3, config.hidden_dim),  # decisions + confidences + specializations
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def assign_roles(self, specializations, task_features, context):
        """Assign roles to agents based on specializations and task requirements"""
        batch_size, num_agents, spec_dim = specializations.shape
        
        # Combine specialization, task features, and context
        combined_features = torch.cat([
            specializations.view(batch_size * num_agents, -1),
            task_features.repeat(num_agents, 1),
            context.repeat(num_agents, 1)
        ], dim=-1)
        
        # Assign roles
        role_probs = self.role_assignment(combined_features)
        role_probs = role_probs.view(batch_size, num_agents, -1)
        
        return role_probs
    
    def fuse_information(self, decisions, confidences, specializations):
        """Fuse information from multiple agents"""
        # Compute weights based on specialization relevance and confidence
        spec_relevance = torch.sum(
            specializations * self.task_embedding.unsqueeze(0).unsqueeze(0), 
            dim=-1
        )
        
        weights = F.softmax(spec_relevance + torch.log(confidences.squeeze(-1) + 1e-8), dim=-1)
        
        # Weighted fusion
        fused_decision = torch.sum(weights.unsqueeze(-1) * decisions, dim=1)
        
        return fused_decision, weights
    
    def resolve_conflicts(self, decisions, confidences, specializations):
        """Resolve conflicts between agent decisions"""
        batch_size, num_agents = decisions.shape[:2]
        
        # Combine all information for arbitration
        arbitration_input = torch.cat([
            decisions.view(batch_size, -1),
            confidences.view(batch_size, -1),
            specializations.view(batch_size, -1)
        ], dim=-1)
        
        # Generate arbitration decision
        arbitration_score = self.arbitration_network(arbitration_input)
        
        return arbitration_score

class CMAN(nn.Module):
    """Complete Collaborative Multimodal Agent Network"""
    
    def __init__(self, config: CMANConfig):
        super().__init__()
        self.config = config
        
        # Create agents
        self.agents = nn.ModuleList([
            Agent(config, i) for i in range(config.num_agents)
        ])
        
        # Global communication module
        self.communication = CommunicationModule(config)
        
        # Coordination module
        self.coordination = DynamicAgentCoordination(config)
        
        # Task-specific output head
        self.output_head = nn.Linear(config.hidden_dim, 1)  # Adjust based on task
        
    def compute_specialization_loss(self, specializations):
        """Compute specialization loss components"""
        batch_size, num_agents, spec_dim = specializations.shape
        
        # Diversity loss
        div_loss = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                div_loss -= torch.mean(torch.norm(
                    specializations[:, i] - specializations[:, j], 
                    p=2, dim=-1
                ) ** 2)
        
        # L1 sparsity penalty
        sparsity_loss = self.config.gamma * torch.mean(torch.norm(specializations, p=1, dim=-1))
        
        # Regularization loss
        reg_loss = torch.mean(torch.clamp(
            torch.norm(specializations, p=float('inf'), dim=-1) - self.config.tau_threshold,
            min=0
        ) ** 2)
        
        return div_loss + sparsity_loss, reg_loss
    
    def compute_communication_loss(self, messages):
        """Compute communication efficiency loss"""
        # L2 penalty for message compactness
        l2_loss = torch.mean(torch.norm(messages, p=2, dim=-1) ** 2)
        
        # KL divergence penalty (simplified)
        message_mean = torch.mean(messages, dim=[0, 1])
        prior_mean = torch.zeros_like(message_mean)
        kl_loss = F.kl_div(
            F.log_softmax(message_mean, dim=-1),
            F.softmax(prior_mean, dim=-1),
            reduction='batchmean'
        )
        
        return l2_loss + self.config.beta_kl * kl_loss
    
    def forward(self, vision_input, audio_input, text_input, return_communication=False):
        batch_size = vision_input.size(0) if vision_input is not None else audio_input.size(0)
        
        # Phase 1: Individual agent processing
        agent_outputs = []
        hidden_states = []
        specializations = []
        
        for agent in self.agents:
            output = agent(vision_input, audio_input, text_input)
            agent_outputs.append(output)
            hidden_states.append(output['hidden_state'])
            specializations.append(output['specialization'])
        
        hidden_states = torch.stack(hidden_states, dim=1)
        specializations = torch.stack(specializations, dim=1)
        
        # Phase 2: Inter-agent communication
        peer_messages, raw_messages = self.communication(hidden_states, specializations)
        
        # Phase 3: Update agent outputs with communication
        updated_outputs = []
        for i, agent in enumerate(self.agents):
            output = agent(
                vision_input, audio_input, text_input, 
                peer_messages=peer_messages[:, i]
            )
            updated_outputs.append(output)
        
        # Extract updated information
        decisions = torch.stack([out['decision'] for out in updated_outputs], dim=1)
        confidences = torch.stack([out['confidence'] for out in updated_outputs], dim=1)
        
        # Phase 4: Dynamic coordination
        task_features = torch.mean(hidden_states, dim=1)  # Simple task representation
        context = torch.mean(hidden_states, dim=1)  # Context representation
        
        # Role assignment
        role_probs = self.coordination.assign_roles(specializations, task_features, context)
        
        # Information fusion
        fused_decision, fusion_weights = self.coordination.fuse_information(
            decisions, confidences, specializations
        )
        
        # Final output
        final_output = self.output_head(fused_decision)
        
        # Compute losses
        div_loss, reg_loss = self.compute_specialization_loss(specializations)
        comm_loss = self.compute_communication_loss(raw_messages)
        
        result = {
            'output': final_output,
            'decisions': decisions,
            'confidences': confidences,
            'specializations': specializations,
            'fusion_weights': fusion_weights,
            'role_probs': role_probs,
            'losses': {
                'diversity': div_loss,
                'regularization': reg_loss,
                'communication': comm_loss
            }
        }
        
        if return_communication:
            result['messages'] = raw_messages
            result['peer_messages'] = peer_messages
        
        return result

class CMANTrainer:
    """Training loop for CMAN"""
    
    def __init__(self, model: CMAN, config: CMANConfig):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.warmup_steps
        )
        
        # Loss tracking
        self.losses = {
            'total': [],
            'task': [],
            'diversity': [],
            'communication': [],
            'regularization': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def compute_total_loss(self, outputs, targets, phase='specialization'):
        """Compute total training loss"""
        # Task loss (MSE for regression, modify for specific tasks)
        task_loss = F.mse_loss(outputs['output'], targets)
        
        if phase == 'pretrain':
            return task_loss
        
        # Specialization losses
        div_loss = outputs['losses']['diversity']
        reg_loss = outputs['losses']['regularization']
        comm_loss = outputs['losses']['communication']
        
        # Combined loss
        total_loss = (
            task_loss + 
            self.config.lambda_comm * comm_loss +
            self.config.lambda_div * div_loss +
            self.config.lambda_reg * reg_loss
        )
        
        return total_loss, {
            'task': task_loss.item(),
            'diversity': div_loss.item(),
            'communication': comm_loss.item(),
            'regularization': reg_loss.item()
        }
    
    def train_epoch(self, dataloader, phase='specialization'):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            vision_input = batch.get('vision', None)
            audio_input = batch.get('audio', None)
            text_input = batch.get('text', None)
            targets = batch['targets'].to(self.device)
            
            if vision_input is not None:
                vision_input = vision_input.to(self.device)
            if audio_input is not None:
                audio_input = audio_input.to(self.device)
            if text_input is not None:
                text_input = text_input.to(self.device)
            
            # Forward pass
            outputs = self.model(vision_input, audio_input, text_input)
            
            # Compute loss
            if phase == 'pretrain':
                loss = self.compute_total_loss(outputs, targets, phase)
                loss_dict = {'task': loss.item()}
            else:
                loss, loss_dict = self.compute_total_loss(outputs, targets, phase)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_losses.append(loss_dict)
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"Task: {loss_dict['task']:.4f}"
                )
        
        return epoch_losses
    
    def train(self, train_dataloader, val_dataloader=None):
        """Full training procedure"""
        # Phase 1: Cooperative pre-training
        self.logger.info("Starting cooperative pre-training phase...")
        for epoch in range(self.config.pretrain_epochs):
            self.logger.info(f"Pre-training epoch {epoch + 1}/{self.config.pretrain_epochs}")
            epoch_losses = self.train_epoch(train_dataloader, phase='pretrain')
            
            # Validation
            if val_dataloader is not None and epoch % 10 == 0:
                val_loss = self.evaluate(val_dataloader)
                self.logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Phase 2: Specialization learning
        self.logger.info("Starting specialization learning phase...")
        for epoch in range(self.config.specialization_epochs):
            self.logger.info(f"Specialization epoch {epoch + 1}/{self.config.specialization_epochs}")
            
            # Gradually increase diversity penalty
            current_lambda_div = min(
                self.config.lambda_div,
                self.config.lambda_div * (epoch + 1) / 25
            )
            self.config.lambda_div = current_lambda_div
            
            epoch_losses = self.train_epoch(train_dataloader, phase='specialization')
            
            # Track losses
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses if key in loss])
                for key in ['task', 'diversity', 'communication', 'regularization']
            }
            
            for key, value in avg_losses.items():
                self.losses[key].append(value)
            
            # Validation
            if val_dataloader is not None and epoch % 10 == 0:
                val_loss = self.evaluate(val_dataloader)
                self.logger.info(f"Validation loss: {val_loss:.4f}")
                self.logger.info(f"Average losses: {avg_losses}")
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                vision_input = batch.get('vision', None)
                audio_input = batch.get('audio', None)
                text_input = batch.get('text', None)
                targets = batch['targets'].to(self.device)
                
                if vision_input is not None:
                    vision_input = vision_input.to(self.device)
                if audio_input is not None:
                    audio_input = audio_input.to(self.device)
                if text_input is not None:
                    text_input = text_input.to(self.device)
                
                outputs = self.model(vision_input, audio_input, text_input)
                loss = F.mse_loss(outputs['output'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

class CMANAnalyzer:
    """Analysis tools for CMAN behavior"""
    
    def __init__(self, model: CMAN):
        self.model = model
        self.model.eval()
    
    def analyze_specializations(self):
        """Analyze agent specializations"""
        specializations = []
        for agent in self.model.agents:
            spec = agent.specialization.detach().cpu().numpy()
            specializations.append(spec)
        
        specializations = np.array(specializations)
        
        # Compute specialization entropy
        entropies = []
        for spec in specializations:
            normalized = F.softmax(torch.tensor(spec), dim=0).numpy()
            entropy = -np.sum(normalized * np.log(normalized + 1e-8))
            entropies.append(entropy)
        
        # Compute inter-agent diversity
        diversity_scores = []
        for i in range(len(specializations)):
            for j in range(i + 1, len(specializations)):
                diversity = np.linalg.norm(specializations[i] - specializations[j])
                diversity_scores.append(diversity)
        
        return {
            'specializations': specializations,
            'entropies': entropies,
            'avg_entropy': np.mean(entropies),
            'diversity_scores': diversity_scores,
            'avg_diversity': np.mean(diversity_scores)
        }
    
    def analyze_communication(self, dataloader, num_samples=100):
        """Analyze communication patterns"""
        messages_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                vision_input = batch.get('vision', None)
                audio_input = batch.get('audio', None)
                text_input = batch.get('text', None)
                
                if vision_input is not None:
                    vision_input = vision_input.to(self.model.config.device)
                if audio_input is not None:
                    audio_input = audio_input.to(self.model.config.device)
                if text_input is not None:
                    text_input = text_input.to(self.model.config.device)
                
                outputs = self.model(
                    vision_input, audio_input, text_input,
                    return_communication=True
                )
                
                messages = outputs['messages'].cpu().numpy()
                messages_list.append(messages)
        
        all_messages = np.concatenate(messages_list, axis=0)
        
        # Reshape for analysis
        batch_size, num_agents, message_dim = all_messages.shape
        messages_flat = all_messages.reshape(-1, message_dim)
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        messages_2d = tsne.fit_transform(messages_flat)
        
        return {
            'messages': all_messages,
            'messages_2d': messages_2d,
            'message_stats': {
                'mean': np.mean(messages_flat, axis=0),
                'std': np.std(messages_flat, axis=0),
                'norm': np.linalg.norm(messages_flat, axis=1)
            }
        }
    
    def visualize_specializations(self, save_path=None):
        """Visualize agent specializations"""
        analysis = self.analyze_specializations()
        specializations = analysis['specializations']
        
        plt.figure(figsize=(12, 8))
        
        # Heatmap of specializations
        plt.subplot(2, 2, 1)
        sns.heatmap(specializations, annot=True, cmap='viridis')
        plt.title('Agent Specializations')
        plt.xlabel('Specialization Dimension')
        plt.ylabel('Agent ID')
        
        # Specialization entropies
        plt.subplot(2, 2, 2)
        plt.bar(range(len(analysis['entropies'])), analysis['entropies'])
        plt.title('Specialization Entropies')
        plt.xlabel('Agent ID')
        plt.ylabel('Entropy')
        
        # Diversity matrix
        plt.subplot(2, 2, 3)
        num_agents = len(specializations)
        diversity_matrix = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    diversity_matrix[i, j] = np.linalg.norm(
                        specializations[i] - specializations[j]
                    )
        
        sns.heatmap(diversity_matrix, annot=True, cmap='coolwarm')
        plt.title('Inter-Agent Diversity Matrix')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')
        
        # Principal components
        plt.subplot(2, 2, 4)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        spec_pca = pca.fit_transform(specializations)
        
        plt.scatter(spec_pca[:, 0], spec_pca[:, 1])
        for i, (x, y) in enumerate(spec_pca):
            plt.annotate(f'Agent {i}', (x, y), xytext=(5, 5), 
                        textcoords='offset points')
        plt.title('Specializations in PCA Space')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_communication(self, dataloader, save_path=None):
        """Visualize communication patterns"""
        analysis = self.analyze_communication(dataloader)
        messages_2d = analysis['messages_2d']
        
        plt.figure(figsize=(15, 10))
        
        # t-SNE visualization
        plt.subplot(2, 3, 1)
        batch_size = analysis['messages'].shape[0]
        num_agents = analysis['messages'].shape[1]
        
        colors = plt.cm.Set1(np.linspace(0, 1, num_agents))
        for agent_id in range(num_agents):
            agent_messages = messages_2d[agent_id::num_agents]
            plt.scatter(agent_messages[:, 0], agent_messages[:, 1], 
                       c=[colors[agent_id]], label=f'Agent {agent_id}', alpha=0.6)
        
        plt.title('Message Embeddings (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')  
        plt.legend()
        
        # Message norms by agent
        plt.subplot(2, 3, 2)
        message_norms = analysis['message_stats']['norm']
        agent_norms = []
        for agent_id in range(num_agents):
            agent_message_norms = message_norms[agent_id::num_agents]
            agent_norms.append(np.mean(agent_message_norms))
        
        plt.bar(range(num_agents), agent_norms)
        plt.title('Average Message Norms by Agent')
        plt.xlabel('Agent ID')
        plt.ylabel('Average Message Norm')
        
        # Message dimension analysis
        plt.subplot(2, 3, 3)
        message_means = analysis['message_stats']['mean']
        message_stds = analysis['message_stats']['std']
        
        plt.errorbar(range(len(message_means)), message_means, yerr=message_stds, 
                    capsize=3, capthick=1)
        plt.title('Message Statistics by Dimension')
        plt.xlabel('Message Dimension')
        plt.ylabel('Mean ± Std')
        
        # Communication matrix (simplified)
        plt.subplot(2, 3, 4)
        # Compute pairwise message similarities
        all_messages = analysis['messages']
        batch_size, num_agents, msg_dim = all_messages.shape
        
        similarity_matrix = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    msg_i = all_messages[:, i].reshape(-1)
                    msg_j = all_messages[:, j].reshape(-1)
                    similarity = np.corrcoef(msg_i, msg_j)[0, 1]
                    similarity_matrix[i, j] = similarity if not np.isnan(similarity) else 0
        
        sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Inter-Agent Message Similarity')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')
        
        # Message complexity over time
        plt.subplot(2, 3, 5)
        message_complexities = []
        for batch_idx in range(min(batch_size, 50)):  # Limit for visualization
            batch_messages = all_messages[batch_idx]
            complexity = np.mean(np.std(batch_messages, axis=1))
            message_complexities.append(complexity)
        
        plt.plot(message_complexities)
        plt.title('Message Complexity Over Time')
        plt.xlabel('Batch Index')
        plt.ylabel('Average Message Std')
        
        # Agent communication frequency
        plt.subplot(2, 3, 6)
        # This is a simplified version - in practice you'd track actual communication
        comm_frequencies = np.random.rand(num_agents, num_agents)  # Placeholder
        np.fill_diagonal(comm_frequencies, 0)
        
        sns.heatmap(comm_frequencies, annot=True, cmap='Blues')
        plt.title('Communication Frequency Matrix')
        plt.xlabel('Receiver Agent')
        plt.ylabel('Sender Agent')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example dataset class for multimodal data
class MultimodalDataset(Dataset):
    """Example dataset class for multimodal inputs"""
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Load your data here - this is a placeholder
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load dataset samples - implement based on your data format"""
        # Placeholder implementation
        samples = []
        for i in range(1000):  # Dummy data
            sample = {
                'vision': torch.randn(3, 224, 224),
                'audio': torch.randn(1, 16000),  # 1 second at 16kHz
                'text': torch.randint(0, 30000, (50,)),  # 50 tokens
                'target': torch.randn(1)
            }
            samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return {
            'vision': sample['vision'],
            'audio': sample['audio'], 
            'text': sample['text'],
            'targets': sample['target']
        }

# Evaluation metrics
class CMANEvaluator:
    """Evaluation metrics for CMAN"""
    
    def __init__(self, model: CMAN):
        self.model = model
        self.model.eval()
    
    def evaluate_task_performance(self, dataloader, task_type='regression'):
        """Evaluate main task performance"""
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                vision_input = batch.get('vision', None)
                audio_input = batch.get('audio', None)
                text_input = batch.get('text', None)
                batch_targets = batch['targets']
                
                if vision_input is not None:
                    vision_input = vision_input.to(self.model.config.device)
                if audio_input is not None:
                    audio_input = audio_input.to(self.model.config.device)
                if text_input is not None:
                    text_input = text_input.to(self.model.config.device)
                
                outputs = self.model(vision_input, audio_input, text_input)
                
                predictions.append(outputs['output'].cpu())
                targets.append(batch_targets)
        
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        if task_type == 'regression':
            mse = F.mse_loss(predictions, targets)
            mae = F.l1_loss(predictions, targets)
            return {'mse': mse.item(), 'mae': mae.item()}
        
        elif task_type == 'classification':
            predictions = torch.argmax(predictions, dim=1)
            targets = torch.argmax(targets, dim=1)
            accuracy = (predictions == targets).float().mean()
            return {'accuracy': accuracy.item()}
    
    def evaluate_robustness(self, dataloader, noise_levels=[0.1, 0.2, 0.3]):
        """Evaluate robustness to input noise"""
        results = {}
        
        for noise_level in noise_levels:
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in dataloader:
                    vision_input = batch.get('vision', None)
                    audio_input = batch.get('audio', None)
                    text_input = batch.get('text', None)
                    batch_targets = batch['targets']
                    
                    # Add noise
                    if vision_input is not None:
                        vision_input = vision_input.to(self.model.config.device)
                        vision_input += torch.randn_like(vision_input) * noise_level
                    
                    if audio_input is not None:
                        audio_input = audio_input.to(self.model.config.device)
                        audio_input += torch.randn_like(audio_input) * noise_level
                    
                    outputs = self.model(vision_input, audio_input, text_input)
                    
                    predictions.append(outputs['output'].cpu())
                    targets.append(batch_targets)
            
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            
            mse = F.mse_loss(predictions, targets)
            results[f'noise_{noise_level}'] = mse.item()
        
        return results
    
    def evaluate_agent_dropout(self, dataloader, dropout_rates=[0.25, 0.5, 0.75]):
        """Evaluate performance with agent dropout"""
        results = {}
        original_agents = list(self.model.agents)
        
        for dropout_rate in dropout_rates:
            num_agents_to_disable = int(len(original_agents) * dropout_rate)
            disabled_indices = np.random.choice(
                len(original_agents), num_agents_to_disable, replace=False
            )
            
            # Temporarily disable agents
            disabled_agents = []
            for idx in disabled_indices:
                disabled_agents.append(self.model.agents[idx])
                self.model.agents[idx] = None
            
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in dataloader:
                    vision_input = batch.get('vision', None)
                    audio_input = batch.get('audio', None)
                    text_input = batch.get('text', None)
                    batch_targets = batch['targets']
                    
                    if vision_input is not None:
                        vision_input = vision_input.to(self.model.config.device)
                    if audio_input is not None:
                        audio_input = audio_input.to(self.model.config.device)
                    if text_input is not None:
                        text_input = text_input.to(self.model.config.device)
                    
                    # Modified forward pass for agent dropout
                    outputs = self._forward_with_dropout(
                        vision_input, audio_input, text_input, disabled_indices
                    )
                    
                    predictions.append(outputs['output'].cpu())
                    targets.append(batch_targets)
            
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            
            mse = F.mse_loss(predictions, targets)
            results[f'dropout_{dropout_rate}'] = mse.item()
            
            # Restore disabled agents
            for i, idx in enumerate(disabled_indices):
                self.model.agents[idx] = disabled_agents[i]
        
        return results
    
    def _forward_with_dropout(self, vision_input, audio_input, text_input, disabled_indices):
        """Modified forward pass with agent dropout"""
        batch_size = vision_input.size(0) if vision_input is not None else audio_input.size(0)
        
        # Phase 1: Individual agent processing (skip disabled agents)
        agent_outputs = []
        hidden_states = []
        specializations = []
        
        for i, agent in enumerate(self.model.agents):
            if i in disabled_indices:
                continue
                
            output = agent(vision_input, audio_input, text_input)
            agent_outputs.append(output)
            hidden_states.append(output['hidden_state'])
            specializations.append(output['specialization'])
        
        if not hidden_states:  # All agents disabled
            return {'output': torch.zeros(batch_size, 1)}
        
        hidden_states = torch.stack(hidden_states, dim=1)
        specializations = torch.stack(specializations, dim=1)
        
        # Rest of forward pass with remaining agents
        peer_messages, raw_messages = self.model.communication(hidden_states, specializations)
        
        # Update outputs
        updated_outputs = []
        for i, (agent, output) in enumerate(zip([a for j, a in enumerate(self.model.agents) if j not in disabled_indices], agent_outputs)):
            updated_output = agent(
                vision_input, audio_input, text_input,
                peer_messages=peer_messages[:, i]
            )
            updated_outputs.append(updated_output)
        
        decisions = torch.stack([out['decision'] for out in updated_outputs], dim=1)
        confidences = torch.stack([out['confidence'] for out in updated_outputs], dim=1)
        
        # Coordination with remaining agents
        task_features = torch.mean(hidden_states, dim=1)
        context = torch.mean(hidden_states, dim=1)
        
        fused_decision, _ = self.model.coordination.fuse_information(
            decisions, confidences, specializations
        )
        
        final_output = self.model.output_head(fused_decision)
        
        return {'output': final_output}

# Main training script
def main():
    """Main training script"""
    # Configuration
    config = CMANConfig(
        num_agents=4,
        hidden_dim=768,
        specialization_dim=64,
        message_dim=256,
        pretrain_epochs=10,  # Reduced for demo
        specialization_epochs=20,  # Reduced for demo
        batch_size=16  # Reduced for demo
    )
    
    # Create model
    model = CMAN(config).to(config.device)
    
    # Create datasets
    train_dataset = MultimodalDataset('train_data_path')
    val_dataset = MultimodalDataset('val_data_path')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Create trainer
    trainer = CMANTrainer(model, config)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate model
    evaluator = CMANEvaluator(model)
    
    # Task performance
    task_results = evaluator.evaluate_task_performance(val_loader)
    print(f"Task Performance: {task_results}")
    
    # Robustness evaluation
    robustness_results = evaluator.evaluate_robustness(val_loader)
    print(f"Robustness Results: {robustness_results}")
    
    # Agent dropout evaluation
    dropout_results = evaluator.evaluate_agent_dropout(val_loader)
    print(f"Agent Dropout Results: {dropout_results}")
    
    # Analysis and visualization
    analyzer = CMANAnalyzer(model)
    
    # Analyze specializations
    spec_analysis = analyzer.analyze_specializations()
    print(f"Specialization Analysis: {spec_analysis}")
    
    # Visualize results
    analyzer.visualize_specializations('specializations.png')
    analyzer.visualize_communication(val_loader, 'communication.png')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_losses': trainer.losses
    }, 'cman_model.pth')
    
    print("Training completed successfully!")

# Utility functions for different benchmarks
class BenchmarkDatasets:
    """Utilities for loading different benchmark datasets"""
    
    @staticmethod
    def load_avsd_complex(data_path):
        """Load AVSD-Complex dataset"""
        # Implement dataset loading logic
        pass
    
    @staticmethod
    def load_multimodal_vqa(data_path):
        """Load MultiModal-VQA dataset"""
        # Implement dataset loading logic
        pass
    
    @staticmethod
    def load_dynamic_scenes(data_path):
        """Load Dynamic-Scenes dataset"""
        # Implement dataset loading logic
        pass
    
    @staticmethod
    def load_ego4d_multimodal(data_path):
        """Load Ego4D-Multimodal dataset"""
        # Implement dataset loading logic
        pass
    
    @staticmethod
    def load_msr_vtt_audio(data_path):
        """Load MSR-VTT-Audio dataset"""
        # Implement dataset loading logic
        pass

# Configuration for different experiments
class ExperimentConfigs:
    """Pre-defined configurations for different experiments"""
    
    @staticmethod
    def get_baseline_config():
        """Configuration for baseline experiments"""
        return CMANConfig(
            num_agents=1,  # Single agent baseline
            lambda_div=0.0,  # No diversity loss
            lambda_comm=0.0  # No communication loss
        )
    
    @staticmethod
    def get_fixed_role_config():
        """Configuration for fixed-role multi-agent baseline"""
        config = CMANConfig()
        config.lambda_div = 0.0  # No dynamic specialization
        return config
    
    @staticmethod
    def get_full_cman_config():
        """Configuration for full CMAN system"""
        return CMANConfig()
    
    @staticmethod
    def get_ablation_configs():
        """Configurations for ablation studies"""
        configs = {}
        
        # No dynamic specialization
        configs['no_dynamic_spec'] = CMANConfig(lambda_div=0.0)
        
        # No emergent communication
        configs['no_emergent_comm'] = CMANConfig(lambda_comm=0.0)
        
        # No DAC algorithm (simplified coordination)
        configs['no_dac'] = CMANConfig()  # Would need code modification
        
        # Different numbers of agents
        for n in [2, 3, 5, 6]:
            configs[f'{n}_agents'] = CMANConfig(num_agents=n)
        
        return configs

# Testing utilities
class CMANTester:
    """Utilities for testing CMAN implementation"""
    
    @staticmethod
    def test_model_forward():
        """Test basic model forward pass"""
        config = CMANConfig(
            num_agents=2,
            hidden_dim=256,
            specialization_dim=32,
            message_dim=128
        )
        
        model = CMAN(config)
        
        # Create dummy inputs
        batch_size = 4
        vision_input = torch.randn(batch_size, 3, 224, 224)
        audio_input = torch.randn(batch_size, 1, 16000)
        text_input = torch.randint(0, 1000, (batch_size, 50))
        
        # Forward pass
        outputs = model(vision_input, audio_input, text_input)
        
        assert outputs['output'].shape == (batch_size, 1)
        assert outputs['decisions'].shape == (batch_size, config.num_agents, 1)
        assert outputs['specializations'].shape == (batch_size, config.num_agents, config.specialization_dim)
        
        print("✓ Model forward pass test passed")
    
    @staticmethod
    def test_communication_module():
        """Test communication module"""
        config = CMANConfig(num_agents=3, hidden_dim=256, message_dim=128)
        comm_module = CommunicationModule(config)
        
        batch_size = 2
        hidden_states = torch.randn(batch_size, config.num_agents, config.hidden_dim)
        specializations = torch.randn(batch_size, config.num_agents, config.specialization_dim)
        
        messages, raw_messages = comm_module(hidden_states, specializations)
        
        assert messages.shape == (batch_size, config.num_agents, config.hidden_dim)
        assert raw_messages.shape == (batch_size, config.num_agents, config.message_dim)
        
        print("✓ Communication module test passed")
    
    @staticmethod
    def test_specialization_dynamics():
        """Test specialization learning"""
        config = CMANConfig(num_agents=3)
        model = CMAN(config)
        
        # Get initial specializations
        initial_specs = []
        for agent in model.agents:
            initial_specs.append(agent.specialization.clone())
        
        # Dummy training step
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        vision_input = torch.randn(2, 3, 224, 224)
        audio_input = torch.randn(2, 1, 16000)
        text_input = torch.randint(0, 1000, (2, 50))
        targets = torch.randn(2, 1)
        
        outputs = model(vision_input, audio_input, text_input)
        
        # Compute loss
        task_loss = F.mse_loss(outputs['output'], targets)
        total_loss = task_loss + config.lambda_div * outputs['losses']['diversity']
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Check if specializations changed
        changed = False
        for i, agent in enumerate(model.agents):
            if not torch.allclose(initial_specs[i], agent.specialization):
                changed = True
                break
        
        assert changed, "Specializations should change during training"
        print("✓ Specialization dynamics test passed")
    
    @staticmethod
    def run_all_tests():
        """Run all tests"""
        print("Running CMAN tests...")
        CMANTester.test_model_forward()
        CMANTester.test_communication_module() 
        CMANTester.test_specialization_dynamics()
        print("All tests passed! ✓")

if __name__ == "__main__":
    # Run tests first
    CMANTester.run_all_tests()
    
    # Run main training
    # main()
