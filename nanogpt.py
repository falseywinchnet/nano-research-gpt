#this file contains an experimental, somewhat slow to train, gpt model containing some experimental features
#copyright joshuah rainstar 2025
#licensed under christian freeware licensefor epoch in range(num_epochs):
    
    model.train()

    total_loss = 0.0
    for step in range(1000):  # Adjust the number of steps as needed.
        x_batch, y_batch = get_batch(batch_size, seq_len)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        with autocast():
            p = model(x_batch)
            if USE_HLOSS:                  
                    
                    #probs = apply_logistic_scaling_with_placeholder(p, placeholder_idx, points=1000, temperature=1.0)
                    #probs_flat = probs.view(-1, vocab_size)  # Now shape is (B*seq_len, vocab_size)
                    loss = -torch.log(torch.gather(p, -1, y_batch.unsqueeze(-1)) + 1e-8).squeeze(-1).mean()
                    #loss = per_token_loss.mean()
            else:
                # Generic GPT (non-harmonic) loss path.
                logits = model(x_batch)  # (B, seq, vocab_size)
                logits_flat = logits.view(-1, vocab_size)
                y_flat = y_batch.view(-1)
                loss = criterion_ce(logits_flat, y_flat)

                
        main_loss = loss.detach()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += main_loss
        losses.append(main_loss.cpu())
        if step % 200 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {main_loss:.4f}")

    print(f"Epoch {epoch+1} Average Loss: {total_loss/10:.4f}")

# ====================================================
# Evaluation: Text Generation
# ====================================================

    # Decay rate (tune this to control how fast the bonus decays)
model.eval()
with torch.no_grad():
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    generated = context
    for _ in range(200):  # Generate 200 tokens.
        inp = generated[:, -seq_len:]
        if USE_HLOSS:
            p = model(inp)  # p: (B, seq, vocab_size)
            last_token_probs = p[:, -1, :]  # Shape: [batch_size, vocab_size]
            next_token = torch.multinomial(last_token_probs, num_samples=1)
        else:
            logits = model(inp)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)

