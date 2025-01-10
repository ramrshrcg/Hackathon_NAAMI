document.getElementById('submitBtn').addEventListener('click', async function() {
    const postContent = document.getElementById('post').value;

    if (!postContent) {
        alert("Please enter a post or status.");
        return;
    }

    try {
        // Send the post to the backend
        const response = await fetch('http://localhost:3000/submit', { // Replace with your backend URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },  
            body: JSON.stringify({ post: postContent })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();

        const likes= data.metrics.likesCount
        const comment= data.metrics.commentsCount
        const share= data.metrics.sharesCount

        const totoal= likes+comment+share

        const likeper=(likes/totoal)*100
        const commentper=(comment/totoal)*100
        const shareper=(share/totoal)*100

        




        // Update progress bars with data from the response
        document.querySelector('#likes div').style.width = `${likeper}% `;
        document.querySelector('#comments div').style.width = `${commentper}%`;
        document.querySelector('#shares div').style.width = `${shareper}%`;

 // Update counts next to progress bars
 document.getElementById('likesCount').innerText = likes || "0";
 document.getElementById('commentsCount').innerText = comment || "0";
 document.getElementById('sharesCount').innerText = share || "0";

        // Generate mock comments
        const mockComments = data.mockComments || [
            'Great post! 😊',
            "I don't agree with this. 😐",
            'This is very insightful! 👍'
        ];
        
        document.getElementById('mock-comments').innerHTML = `
            <h3>Comments</h3>
            ${mockComments.map(comment => `<span>${comment}</span>`).join('')}
        `;

        // Update emojis based on likes
        const emojiDiv = document.getElementById('emojis');
        emojiDiv.innerHTML = data.metrics.likesCount > data.metrics.comments ? '😍' : data.metrics.commentsCount > data.metrics.sharesCount ? '🙂' : '😔';

    } catch (error) {
        console.error('Error:', error);
    }
});
