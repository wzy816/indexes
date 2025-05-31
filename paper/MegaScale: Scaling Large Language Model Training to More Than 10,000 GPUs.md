# MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs

##

1. **问题解决**：
   这篇论文介绍了一个名为 MegaScale 的生产系统，旨在解决在超过 10,000 个 GPU 上训练大型语言模型（LLMs）时面临的前所未有的挑战。这些挑战包括训练效率和稳定性，因为大规模训练任务的长时间性质使得稳定性成为一个重要的考虑因素。

2. **相关研究**：
   论文提到了多方面的相关工作，包括大型语言模型（LLMs）的训练、模型性能比较、系统基础设施的具体细节、模型优化技术（如稀疏注意力机制、新架构设计、通信加速技术）、数据中心的诊断工具、以及大规模分布式系统的容错技术。

3. **问题解决方式**：

   - **全栈方法**：论文采用了算法和系统组件的全栈协同设计方法，包括模型块和优化器设计、计算与通信重叠、操作符优化、数据流水线和网络性能调整。
   - **混合并行策略**：结合了数据并行、流水线并行、张量并行和序列并行。
   - **深度可观测性**：开发了一套诊断工具，通过深入监控系统组件和事件来识别根本原因，并实现容错和减轻落后者影响。
   - **优化技术**：包括模型架构的修改、通信重叠技术、高效的操作符、数据流水线优化和集体通信组初始化的改进。

4. **实验**：

   - **性能测试**：在不同规模的 GPU 上测试 MegaScale 和 Megatron-LM 的训练效率，包括弱扩展性和强扩展性测试。
   - **模型收敛和稳定性测试**：通过微观基准实验验证算法技术不影响模型收敛，并在真实的生产环境中测试模型的收敛和稳定性。
   - **故障排除**：分析了生产训练作业的故障记录，诊断并修复了特定问题，如计算落后者、MFU 下降问题和网络接口闪烁问题。

5. **进一步探索点**：

   - **优化算法**：进一步探索和开发更高效的训练算法，以提高模型在大规模 GPU 上的收敛速度和稳定性。
   - **系统稳定性**：研究如何更有效地预测和处理大规模分布式系统中可能出现的硬件和软件故障。
   - **资源调度**：改进资源调度策略，以更高效地利用大规模 GPU 集群。
   - **网络优化**：继续研究和开发减少网络拥塞和提高数据传输效率的方法。

6. **主要内容总结**：
   论文详细介绍了 MegaScale 系统的设计、实现和部署经验，这是一个为在超过 10,000 个 GPU 上训练大型语言模型而构建的生产级系统。MegaScale 通过算法和系统的共同设计优化了训练效率，并实现了 55.2%的模型 FLOPs 利用率（MFU），比 Megatron-LM 提高了 1.34 倍。论文还强调了在整个训练过程中容错的重要性，并实现了一个定制的健壮训练框架，以自动定位和修复故障。此外，论文提供了一套全面的监控工具，以便深入观察系统组件和事件，从而有助于识别复杂异常的根本原因。作者认为，这项工作不仅为从事 LLM 训练的工作者提供了实际见解，也为这一快速发展领域的未来研究铺平了道路。

## Summary

The article presents the design, implementation and engineering experience of MegaScale, a production system for training large language models (LLMs) at the scale of more than 10,000 GPUs. The authors discuss the challenges of achieving high training efficiency and stability at this unprecedented scale, and the principles of algorithm-system co-design and in-depth observability that guided the development of MegaScale.

## Key Points

- MegaScale enables training LLMs at the scale of over 10,000 GPUs.
- Achieving high training efficiency and stability at this scale is challenging due to communication overhead, failures, and stragglers.
- MegaScale applies algorithm-system co-design and in-depth observability principles to address these challenges.
- MegaScale incorporates algorithmic optimizations such as parallel transformer block, sliding window attention, and LAMB optimizer to improve training efficiency.
- MegaScale leverages overlapping techniques to hide the communication overhead for data parallelism, pipeline parallelism, and tensor parallelism.
- MegaScale develops a robust training framework and customized diagnosis tools to achieve fault tolerance and mitigate stragglers.
- MegaScale achieves up to 1.34x speedup over the state-of-the-art Megatron-LM framework on training a 175B model, and maintains model convergence and stability in real-world production runs.

## What are the key algorithmic and system-level optimizations incorporated in MegaScale?

The key algorithmic and system-level optimizations incorporated in MegaScale include:

Algorithmic Optimizations:

1. Parallel Transformer Block: Reformatting the transformer block to execute the attention and MLP blocks in parallel, reducing computation time.
2. Sliding Window Attention: Using a fixed-size attention window to reduce the computational complexity compared to full self-attention.
3. LAMB Optimizer: Enabling scaling of the training batch size up to 4x without compromising accuracy.

System-level Optimizations:

1. Communication Overlapping:
   - For data parallelism, overlapping all-gather and reduce-scatter operations with computation.
   - For pipeline parallelism, decoupling send and receive operations to overlap with computation.
   - For tensor/sequence parallelism, fusing communication operations with compute-intensive GEMM kernels.
2. Efficient Operator Implementations:
   - Optimizing attention, LayerNorm, and GELU operators for better performance.
3. Data Pipeline Optimizations:
   - Asynchronous data preprocessing to hide preprocessing overhead.
   - Eliminating redundant data loaders by using a shared memory approach.
4. Collective Communication Group Initialization Optimizations:
   - Replacing the default synchronous barrier with an asynchronous, non-blocking approach.
   - Minimizing the need for global barriers by carefully designing the order of communication group initialization.
5. Network Performance Tuning:
   - Designing a custom network topology to reduce ECMP hashing conflicts.
   - Developing a congestion control algorithm that integrates DCQCN and Swift principles.
   - Tuning retransmit timeout parameters for faster recovery from link flapping.

These optimizations at both the algorithmic and system levels enable MegaScale to achieve significantly higher training efficiency and stability compared to the state-of-the-art Megatron-LM framework.

## 并行化策略

在论文 "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs" 中，提到了几种不同的并行化策略（parallelism strategies），这些策略被用来扩展大型语言模型（LLMs）的训练，使其能够跨过数以万计的 GPU 进行。以下是对这些并行化策略的解释：

1. **数据并行（Data Parallelism）**:
   数据并行是一种常见的并行化方法，它涉及将数据集分割成多个小块，并在多个设备（如 GPU）上同时处理这些数据块。在 LLMs 训练中，模型和优化器的状态会在所有设备上复制，然后每个设备独立地执行前向和后向传播。在每次迭代后，所有设备会同步它们的模型更新。

2. **模型并行（Model Parallelism）**:
   模型并行涉及将模型的不同部分分布到不同的设备上。这意味着每个设备只负责模型的一部分。这种策略对于非常大的模型特别有用，因为它们可以被分解，使得单个设备不需要存储整个模型。

3. **流水线并行（Pipeline Parallelism）**:
   流水线并行是一种将模型层分配到多个设备上的技术，每个设备拥有模型的一部分。同时，每个训练批次被细分为多个更小的批次（称为 micro-batches），这些小批次以流水线的方式执行。这种策略可以减少训练过程中的空闲时间，提高资源利用率。

4. **张量并行（Tensor Parallelism）**:
   张量并行是将模型中的单个操作符分布到多个设备上，每个设备并行执行计算的一部分。这通常用于计算密集型操作，如矩阵乘法（GEMM）。张量并行可能需要设备之间的通信，以分割输入和合并输出。

5. **序列并行（Sequence Parallelism）**:
   序列并行是一种沿着序列维度分布操作的技术，这可以有效地减少激活内存占用。它通常与张量并行结合使用，以进一步提高训练的扩展性。

6. **3D 并行（3D Parallelism）**:
   3D 并行是一种综合了数据并行、流水线并行和张量/序列并行的策略。这种策略允许在大量 GPU 上扩展 LLMs 的训练，通过在不同的维度上应用并行化来最大化资源利用率和计算效率。

论文中提到，为了最大化性能，他们采用了混合并行策略，并设计了专门的技术来优化每种并行策略的通信和计算重叠。例如，他们利用了数据并行、流水线并行、张量并行和序列并行的组合，并针对每种策略设计了自定义技术，以最大化通信和计算之间的重叠。此外，他们还优化了数据流水线，采用了预取和基于树的加载技术，以及非阻塞异步操作，以提高大规模集体通信的性能。

## all-gather 和 reduce-scatter 算子

在并行计算中，特别是在数据并行训练大型神经网络模型时，all-gather 和 reduce-scatter 是两种常用的通信算子，它们在多 GPU 训练中扮演着重要角色。下面是每个算子的解释：

All-Gather 算子：
All-Gather 是一种通信操作，用于在多个进程（或 GPU）之间收集数据。具体来说，每个进程都有一个数据段，All-Gather 操作会将所有进程中的数据段收集到每个进程中。这意味着，如果在 n 个进程中每个进程都有数据段 A_i，执行 All-Gather 操作后，每个进程都将得到包含所有 A_i 的数据集合。

在深度学习训练中，All-Gather 通常用于数据并行中同步模型参数。例如，在训练开始前，所有 GPU 需要有相同的模型参数副本，All-Gather 操作可以确保这一点。

Reduce-Scatter 算子：
Reduce-Scatter 是一种更复杂的通信操作，它结合了数据的归约（Reduce）和分散（Scatter）两个步骤。在 Reduce-Scatter 操作中，首先所有进程会对自己的输入数据段执行某种归约操作（如求和、求平均等），然后这些归约后的结果会被分散到所有进程中。

在神经网络训练的上下文中，Reduce-Scatter 通常用于反向传播阶段。当每个 GPU 计算出梯度后，需要将这些梯度聚合起来以更新模型参数。Reduce-Scatter 操作会将所有 GPU 的梯度求和（归约），然后将总和分散到每个 GPU（Scatter），这样每个 GPU 都可以用相同的全局梯度来更新参数。

这两种算子是分布式机器学习框架中实现模型参数同步的关键部分，它们确保了在多个 GPU 上训练时，每个 GPU 都能获得全局一致的信息，从而进行正确的模型更新。在大规模训练任务中，高效地执行这些通信操作对于训练的效率和稳定性至关重要。论文中提到的 MegaScale 系统通过优化这些通信操作，提高了模型训练的效率和稳定性。

## pytorch FSDP

PyTorch FSDP，即 Fully Sharded Data Parallelism，是 PyTorch 深度学习框架中的一个高级 API，用于实现数据并行训练，特别是针对大规模模型和数据集。FSDP 旨在提高大规模训练任务的效率和可扩展性，它通过以下方式来优化性能：

1. **模型分片（Model Sharding）**: FSDP 将模型的不同部分（称为 shards）分布到不同的 GPU 上。这意味着每个 GPU 只存储和计算模型的一部分，而不是整个模型。这样可以减少单个 GPU 的内存占用，允许训练更大的模型。

2. **梯度聚合（Gradient Aggregation）**: 在数据并行训练中，通常需要在多个进程间同步梯度。FSDP 通过优化梯度的聚合过程来减少通信开销，特别是在使用大量 GPU 时。

3. **减少内存使用**: 由于模型被分割到不同的 GPU 上，FSDP 可以显著减少总体的内存占用，这对于训练大型模型尤其重要。

4. **动态调整（Dynamic Adjustment）**: FSDP 可以在训练过程中动态地调整模型的分片，以适应不同的训练阶段和性能要求。

5. **兼容性**: FSDP 与 PyTorch 的其它组件和 API 保持兼容，使得开发者可以无缝地将其集成到现有的训练流程中。

在论文 "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs" 中，作者提到了 PyTorch FSDP，并且受到了其启发，在 MegaScale 系统中实现了类似的功能，以优化数据并行训练的性能。具体来说，MegaScale 系统中采用了预取（pre-fetching）技术，允许某些通信操作与数据加载操作并行进行，从而减少了通信时间。此外，MegaScale 还通过优先启动高优先级的通信操作，进一步提高了通信效率。这些优化使得 MegaScale 能够有效地扩展到超过 10,000 个 GPU 的规模，同时保持高效率和稳定性。

## 1F1B Scheduling

1F1B scheduling（One Forward and One Backward scheduling）是一种流水线并行（pipeline parallelism）中的调度策略，用于提高深度学习模型训练的效率。这种策略特别适用于大型模型，如大型语言模型（LLMs），可以在多个 GPU 上并行地执行模型的不同部分。

在 1F1B 调度中，模型被分成多个阶段（或称为“阶段”或“虚拟阶段”），每个阶段处理模型的一部分。具体来说：

- **1F（One Forward）**: 在每个训练批次开始时，输入数据首先进入流水线的第一个阶段。然后，数据依次通过每个阶段，每个阶段执行模型的一部分前向传播（forward pass）。随着数据在流水线中的流动，每个阶段逐步完成其前向传播任务。

- **1B（One Backward）**: 一旦数据通过所有阶段完成前向传播，得到的输出将反向传递回各个阶段，进行反向传播（backward pass）。在 1F1B 调度中，每个阶段在完成其前向传播后，会立即开始执行反向传播。

1F1B 调度的关键优势在于它减少了流水线中的空闲时间（也称为“bubbles”），从而更高效地利用计算资源。在这种策略下，每个阶段在完成一个微批次（micro-batch）的前向传播后，可以立即开始处理下一个微批次的前向传播，而不需要等待所有微批次的前向传播完成。这样可以保持流水线的连续流动，减少等待时间，提高整体的训练吞吐量。

在论文 "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs" 中，作者提到了 Megatron-LM 采用了 1F1B 调度策略，并且 MegaScale 系统也利用了这种策略来优化大规模 GPU 集群上的 LLM 训练。通过精心设计的流水线阶段和通信重叠技术，MegaScale 能够在保持高训练效率的同时，扩展到超过 10,000 个 GPU。

## 3.5 如何优化大规模分布式训练中的集体通信初始化初始化时间

在论文 "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs" 的第 3.5 节中，作者们讨论了如何优化大规模分布式训练中的集体通信初始化（Collective Communication Group Initialization）。以下是对这一部分内容的总结：

**集体通信初始化的挑战**：

- 当 GPU 数量扩展到数千个时，使用默认的 PyTorch 分布式通信库（torch.distributed）初始化 NCCL（NVIDIA Collective Communications Library）通信组的开销变得不可接受。
- 初始化时间随着 GPU 数量的增加而显著增长，这会影响常规测试和迭代开发，也会妨碍快速重启和恢复机制的实施。

**优化策略**：

1. **替换同步机制**：作者们通过分析 torch.distributed，发现初始化时间的一个主要原因是每个进程在初始化特定通信组后都会进行同步操作。默认的同步机制使用 TCPStore，这是一个单线程、阻塞式的读写操作。作者们将 TCPStore 替换为 Redis，Redis 是非阻塞和异步的，从而显著减少了初始化时间。

2. **减少全局屏障的使用**：每个进程在初始化其通信组后执行全局屏障操作。作者们通过精心设计通信组的初始化顺序，减少了全局屏障的需求，将全局屏障的时间复杂度从 O(n^2) 降低到 O(n)。

**实验结果**：

- 通过这些优化，初始化时间在 2048 个 GPU 上减少到 5 秒以下，在超过 10,000 个 GPU 上减少到 30 秒以下。

**结论**：

- 这些优化使得在大规模 GPU 集群上进行 LLM 训练的初始化过程更加高效，减少了训练作业的启动时间，并提高了系统的整体性能。

## LAMB

LAMB（Layer-wise Adaptive Moments optimizer for Batch training）是一种优化算法，专为大型批量训练设计，以提高深度学习模型的训练效率和稳定性。LAMB 结合了 Adam 优化器的自适应学习率特性和批量训练的优势，允许使用更大的批量大小而不损害模型的收敛性。

在深度学习中，使用较大的批量大小可以提高计算资源的利用率，加快训练速度，但同时也可能导致模型训练不收敛或收敛到次优解。Adam 优化器通过维护每个参数的一阶矩（均值）和二阶矩（方差）来调整学习率，从而实现自适应的优化。然而，Adam 在大规模批量训练中也面临着一些问题。

LAMB 通过以下方式改进了 Adam 优化器：

1. **层级自适应学习率**：LAMB 为模型的每一层独立计算自适应学习率，而不是全局共享一个学习率。
2. **批量大小无关性**：LAMB 设计为与批量大小无关，这意味着即使批量大小增加，也能保证模型的收敛性。
3. **权重衰减**：LAMB 可以与权重衰减（weight decay）一起使用，而不会损害优化过程。
4. **更好的泛化性**：LAMB 有助于模型在大规模数据集上训练时获得更好的泛化性能。

在论文 "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs" 中，LAMB 被用来支持大型语言模型（LLMs）的训练，使得训练过程可以扩展到更大的批量大小，同时保持模型的收敛性和性能。这在大规模分布式训练环境中尤其重要，因为它允许使用更多的 GPU 来加速训练过程。[^1^]

[^1^]: Y. You, J. Li, S. Reddi, J. Hseu, S. Kumar, S. Bhojanapalli, X. Song, J. Demmel, K. Keutzer, and C.-J. Hsieh, “Large batch optimization for deep learning: Training BERT in 76 minutes,” in International Conference on Learning Representations, 2020.
