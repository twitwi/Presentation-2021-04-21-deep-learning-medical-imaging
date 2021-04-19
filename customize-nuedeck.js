

function nuedeckAddPlugins(Vue, plugins) {
    Vue.component('w', {
        props: ['href'],
        template: `
        <v-template>
        <slot></slot>
        <a :href="'https://en.wikipedia.org/wiki/' + href" target="_blank" class="wikipedia-link">W</a>
        </v-template>`
    })
    Vue.component('attribute', {
        props: ['src', 'href', 'content'],
        template: `
        <a class="attribute" :href="href" target="_blank">
        <div :style="'background-image: url('+src+')'"></div>
        <div>{{content}}<slot></slot></div>
        </a>`
    })
    Vue.component('help', {
        props: [],
        template: `
        <div class="floating-help">
        <slot></slot>
        </div>`
    })
    Vue.component('note', {
        props: [],
        template: `
        <div class="comment note">
        <slot></slot>
        </div>`
    })
    plugins.push({
        name: 'ClickFS',
        /*async*/ enrichSlideDeck(slides) {
            for (let s of slides) {
                s.contentElement.querySelectorAll('img:not(.noFS), svg:not(.noFS)').forEach(im => {
                    im.setAttribute('onclick', 'if (event.ctrlKey) this.classList.toggle("FS")') // works as it is saved in the template
                })
            }
        }
    })
}

